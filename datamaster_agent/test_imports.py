"""Quick import smoke test — delete after use."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()
print("1. dotenv loaded")
from core.state import AgentState, ProspectContext
print("2. state imported")
from core.prompts import COORDINATOR_SYSTEM, DATA_ARCHITECT_SYSTEM, VALUE_SELLING_SYSTEM, DOCUMENT_ENGINEER_SYSTEM
print("3. prompts imported")
from tools.postgres_rag import similarity_search, ingest_document, multi_query_search
print("4. postgres_rag imported")
from core.memory import get_checkpointer, make_thread_id
print("5. memory imported")
from agents.coordinator import coordinator_node
print("6. coordinator imported")
from agents.data_architect import data_architect_node
print("7. data_architect imported")
from agents.value_selling import value_selling_node
print("8. value_selling imported")
from agents.document_engineer import document_engineer_node
print("9. document_engineer imported")
from tools.deployment import deploy_node
print("10. deployment imported")
from core.graph import get_compiled_graph
print("11. graph imported")
print("\nAll imports OK!")
