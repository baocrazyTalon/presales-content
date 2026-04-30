"""
tools/deployment.py
────────────────────
Deployment node and helpers:
  - deploy_node: LangGraph node that writes the HTML to GitHub and triggers Vercel.
  - _commit_to_github: Creates/updates a file in the target repo via PyGithub.
  - _trigger_vercel_deploy: Calls the Vercel Deploy Hook or REST API.
"""

from __future__ import annotations

import logging
import os
from pathlib import PurePosixPath

from langchain_core.messages import AIMessage

from core.state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GitHub helper
# ─────────────────────────────────────────────────────────────

def _commit_to_github(filename: str, html_content: str) -> str:
    """
    Create or update a file in the GitHub repo.

    Returns the URL of the created/updated file on GitHub.
    """
    from github import Github, GithubException

    token = os.getenv("GITHUB_TOKEN")
    repo_name = os.getenv("GITHUB_REPO")
    if not token or not repo_name:
        raise EnvironmentError(
            "GITHUB_TOKEN and GITHUB_REPO must be set in .env for deployment"
        )
    branch = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

    gh = Github(token)
    repo = gh.get_repo(repo_name)

    # Place generated docs in CLIENTS/ to match existing workspace convention
    file_path = str(PurePosixPath("CLIENTS") / "GENERATED" / filename)
    commit_message = f"chore: add generated presales doc {filename}"

    try:
        existing = repo.get_contents(file_path, ref=branch)
        repo.update_file(
            path=file_path,
            message=commit_message,
            content=html_content,
            sha=existing.sha,  # type: ignore[union-attr]
            branch=branch,
        )
        logger.info("[GitHub] Updated %s on branch %s", file_path, branch)
    except GithubException as exc:
        if exc.status == 404:
            repo.create_file(
                path=file_path,
                message=commit_message,
                content=html_content,
                branch=branch,
            )
            logger.info("[GitHub] Created %s on branch %s", file_path, branch)
        else:
            raise

    return f"https://github.com/{repo_name}/blob/{branch}/{file_path}"


# ─────────────────────────────────────────────────────────────
# Vercel helper
# ─────────────────────────────────────────────────────────────

def _trigger_vercel_deploy() -> str:
    """
    Trigger a Vercel deployment via the Deployments API.

    Returns the deployment URL.
    """
    import httpx

    token = os.getenv("VERCEL_TOKEN")
    project_id = os.getenv("VERCEL_PROJECT_ID")
    if not token or not project_id:
        raise EnvironmentError(
            "VERCEL_TOKEN and VERCEL_PROJECT_ID must be set in .env for deployment"
        )
    team_id = os.getenv("VERCEL_TEAM_ID", "")

    headers = {"Authorization": f"Bearer {token}"}
    params: dict[str, str] = {}
    if team_id:
        params["teamId"] = team_id

    # Create a deployment (Vercel picks up the latest commit automatically
    # when gitSource is configured on the project)
    resp = httpx.post(
        f"https://api.vercel.com/v13/deployments",
        headers=headers,
        params=params,
        json={
            "name": project_id,
            "gitSource": {
                "type": "github",
                "ref": os.getenv("GITHUB_DEFAULT_BRANCH", "main"),
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    deploy_url = f"https://{data.get('url', project_id + '.vercel.app')}"
    logger.info("[Vercel] Deployment triggered: %s", deploy_url)
    return deploy_url


# ─────────────────────────────────────────────────────────────
# LangGraph node
# ─────────────────────────────────────────────────────────────

def deploy_node(state: AgentState) -> AgentState:
    """
    LangGraph node: Deploy.

    1. Commits the generated HTML to GitHub.
    2. Triggers a Vercel deployment.
    3. Updates state with the resulting URLs.
    """
    html = state.get("html_output", "")
    filename = state.get("output_filename", "presales.html")

    if not html:
        logger.error("[Deploy] No HTML content to deploy")
        return {**state, "error": "No HTML content to deploy"}

    try:
        github_url = _commit_to_github(filename, html)
    except Exception as exc:
        logger.error("[Deploy] GitHub commit failed: %s", exc)
        return {**state, "error": f"GitHub error: {exc}"}

    vercel_url = None
    try:
        vercel_url = _trigger_vercel_deploy()
    except Exception as exc:
        logger.warning("[Deploy] Vercel trigger failed (non-fatal): %s", exc)

    logger.info("[Deploy] Done. GitHub=%s Vercel=%s", github_url, vercel_url)

    return {
        **state,
        "github_pr_url": github_url,
        "vercel_url": vercel_url,
        "current_agent": "deploy",
        "messages": [
            AIMessage(
                content=(
                    f"Deployment complete!\n"
                    f"- GitHub: {github_url}\n"
                    f"- Vercel: {vercel_url}"
                )
            )
        ],
    }
