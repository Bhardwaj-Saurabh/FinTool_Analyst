"""
Agent Coordinator Module - Complete Financial Agent with Modular Architecture

This module provides the complete financial agent functionality with intelligent routing,
tool coordination, and backward compatibility. It replaces both modern_financial_agent.py
and financial_agent.py by providing all functionality in a single coordinated system.

Learning Objectives:
- Understand multi-tool coordination and intelligent routing
- Implement LLM-based decision making for tool selection
- Learn result synthesis from multiple data sources
- Build modular agent architecture
- Master PII protection in agent workflows

Your Task: Complete the missing implementations marked with YOUR CODE HERE

Key Features:
- Multi-tool coordination with intelligent routing
- Document analysis (10-K filings) for Apple, Google, Tesla
- Database queries with SQL auto-generation and PII protection
- Real-time market data from Yahoo Finance
- Complete backward compatibility for existing notebooks
- Modular architecture using helper modules
"""

import os
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Complete Financial Agent with Dynamic Multi-Tool Coordination
    
    This class combines the functionality of the original modern_financial_agent.py
    and financial_agent.py into a single coordinated system using modular architecture.
    
    Architecture:
    - Document Tools (3): Individual SEC 10-K filing analysis for Apple, Google, Tesla
    - Function Tools (3): Database SQL queries, real-time market data, PII protection
    - Intelligent Routing: LLM-based tool selection and result synthesis
    - Backward Compatibility: Works with existing notebooks and code
    """
    
    def __init__(self, companies: List[str] = None, verbose: bool = False):
        """
        Initialize the complete financial agent with modular architecture.
        
        Args:
            companies: List of company symbols (default: ["AAPL", "GOOGL", "TSLA"])
            verbose: Whether to show detailed operation information
        """
        self.companies = companies if companies is not None else ["AAPL", "GOOGL", "TSLA"]
        self.verbose = verbose
        self.project_root = Path.cwd()  # Use current working directory
        
        # Company metadata
        self.company_info = {
            "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
            "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology"},
            "TSLA": {"name": "Tesla Inc.", "sector": "Automotive"}
        }
        
        # Storage for tools and engines
        self.document_tools = []
        self.function_tools = []
        self.llm = None
    
        
        self._configure_settings()
        
        # Don't auto-initialize tools - create them lazily when first needed
        self._tools_initialized = False
        
        if self.verbose:
            print("✅ Financial Agent Coordinator Initialized")
            print(f"   Companies: {self.companies}")
            print(f"   Tools will be created automatically when first query is made")
    
  
    def _configure_settings(self):
        """Configure LlamaIndex settings with Vocareum API compatibility

        TODO: Set up the LLM and embedding model for intelligent routing

        Requirements:
        - Create OpenAI LLM with "gpt-3.5-turbo" model and temperature=0
        - Create OpenAIEmbedding with "text-embedding-ada-002" model
        - Use api_base parameter for Vocareum API compatibility (both models)
        - Set Settings.llm and Settings.embed_model
        - Store LLM reference in self.llm for routing decisions

        IMPORTANT NOTE FOR VOCAREUM:
        LlamaIndex requires the api_base parameter to work with Vocareum's OpenAI endpoint.
        Get the base URL from environment: os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        Pass it as api_base parameter to both OpenAI() and OpenAIEmbedding() constructors.
        """
        # Get the Vocareum API base URL from environment
        api_base = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")

        # Create OpenAI LLM with gpt-3.5-turbo and temperature=0
        self.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_base=api_base
        )

        # Create OpenAIEmbedding with text-embedding-ada-002
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_base=api_base
        )

        # Set Settings.llm and Settings.embed_model
        Settings.llm = self.llm
        Settings.embed_model = embed_model

        if self.verbose:
            print("✅ LlamaIndex settings configured with OpenAI models")
    
    
    def setup(self, document_tools: List = None, function_tools: List = None):
        """
        Setup all components using the modular architecture.
        
        Args:
            document_tools: Optional pre-created document tools
            function_tools: Optional pre-created function tools
            
        This method initializes all tools and sets up the routing system.
        If tools are not provided, they will be created automatically.
        """
        if self.verbose:
            print("🔧 Setting up Advanced Financial Agent (Modular Architecture)...")
        
        try:
            if document_tools is not None and function_tools is not None:
                # Use provided tools
                self.document_tools = document_tools
                self.function_tools = function_tools
            else:
                # Create tools automatically
                self._create_tools()
            
            if self.verbose:
                status = self.get_status()
                print(f"✅ Setup complete: {status['document_tools']} document tools, {status['function_tools']} function tools")
                print(f"🎯 System ready: {'✅' if status['ready'] else '❌'}")
                
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if self.verbose:
                print(f"❌ Setup failed: {e}")
    
    def _create_tools(self):
        """Create all tools automatically using helper modules

        TODO: Import and use the DocumentToolsManager and FunctionToolsManager
        to create all necessary tools for the coordinator.

        Steps:
        1. Import DocumentToolsManager from .document_tools
        2. Import FunctionToolsManager from .function_tools
        3. Create instances and call their build methods
        4. Store results in self.document_tools and self.function_tools
        """
        # Import the tool managers
        from .document_tools import DocumentToolsManager
        from .function_tools import FunctionToolsManager

        if self.verbose:
            print("🔧 Creating document tools...")

        # Create DocumentToolsManager instance and build document tools
        doc_manager = DocumentToolsManager(companies=self.companies, verbose=self.verbose)
        self.document_tools = doc_manager.build_document_tools()

        if self.verbose:
            print(f"✅ Created {len(self.document_tools)} document tools")
            print("🔧 Creating function tools...")

        # Create FunctionToolsManager instance and build function tools
        func_manager = FunctionToolsManager(verbose=self.verbose)
        self.function_tools = func_manager.create_function_tools()

        if self.verbose:
            print(f"✅ Created {len(self.function_tools)} function tools")

        # Mark tools as initialized
        self._tools_initialized = True
    
    def _check_and_apply_pii_protection(self, tool_name: str, result: str) -> str:
        """Check if database results need PII protection and apply it automatically
        
        This method automatically detects when database queries return sensitive information
        and applies appropriate PII protection using the PII protection tool from function_tools.
        
        Args:
            tool_name: Name of the tool that generated the result
            result: Raw result string from the tool
            
        Returns:
            Protected result string with PII masked if necessary
        """
        
        # Only apply to database query results
        if "database_query_tool" not in tool_name:
            return result

        # Check if result contains column information (looking for pipe-separated data)
        if "|" not in result:
            return result

        # Extract column names from result
        lines = result.split('\n')
        column_names = []

        for line in lines:
            # Find the header line (contains column names separated by |)
            if '|' in line and not line.startswith('-') and not line.startswith('='):
                # This might be the header line
                parts = [p.strip() for p in line.split('|')]
                # Check if it looks like column names (no numbers or SQL keywords in first part)
                if parts and not any(char.isdigit() for char in parts[0][:5]) and 'SQL' not in parts[0]:
                    column_names = parts
                    break

        if not column_names:
            return result

        # Detect PII fields using _detect_pii_fields()
        pii_fields = self._detect_pii_fields(column_names)

        if not pii_fields:
            return result

        # Find and use the pii_protection_tool
        pii_tool = None
        for tool in self.function_tools:
            if hasattr(tool, 'metadata') and 'pii' in tool.metadata.name.lower():
                pii_tool = tool
                break

        if pii_tool is None:
            return result

        # Apply protection and return masked result
        try:
            # Convert column names to string format for the PII tool
            column_names_str = str(column_names)
            protected_result = pii_tool.call(result, column_names_str)

            # Extract content from ToolOutput if needed
            if hasattr(protected_result, 'content'):
                return str(protected_result.content)
            return str(protected_result)

        except Exception as e:
            if self.verbose:
                print(f"⚠️  PII protection failed: {e}")
            return result
    
    def _detect_pii_fields(self, field_names: list) -> set:
        """Detect which fields contain PII based on field names

        This method identifies potentially sensitive database fields that need protection.

        Args:
            field_names: List of database column names

        Returns:
            Set of field names that contain PII
        """
        # Define PII field patterns
        pii_patterns = [
            'email', 'phone', 'first_name', 'last_name', 'name',
            'address', 'ssn', 'social_security', 'birth_date', 'dob',
            'passport', 'license', 'credit_card', 'account_number'
        ]

        detected_pii = set()

        # Check each field name against patterns
        for field in field_names:
            field_lower = field.lower()
            for pattern in pii_patterns:
                if pattern in field_lower:
                    detected_pii.add(field)
                    break

        return detected_pii
    

    def _route_query(self, query: str) -> List[Tuple[str, str, Any]]:
        """Use LLM to intelligently route query to appropriate tools
        
        This method analyzes the user's query and determines which tools are needed
        to provide a complete answer, then executes those tools and returns results.
        
        Args:
            query: User's natural language query
            
        Returns:
            List of tuples: (tool_name, tool_description, result)
        """
        
        # 1. Create descriptions of all available tools
        all_tools = self.document_tools + self.function_tools
        tool_descriptions = []

        for i, tool in enumerate(all_tools):
            tool_name = tool.metadata.name if hasattr(tool, 'metadata') else f"tool_{i}"
            tool_desc = tool.metadata.description if hasattr(tool, 'metadata') else "No description"
            tool_descriptions.append(f"{i}. {tool_name}: {tool_desc}")

        tools_text = "\n".join(tool_descriptions)

        # 2. Build LLM prompt with query and tool options
        prompt = f"""You are an intelligent tool router for a financial analysis system.
Analyze the user's query and select the most appropriate tool(s) to answer it.

Available Tools:
{tools_text}

User Query: {query}

Routing Guidelines:
- Use document tools (0-2) for questions about company business, strategy, risks, or 10-K filing information
- Use database_query_tool for questions about customers, portfolios, holdings, or investment positions
- Use finance_market_search_tool for current stock prices, market data, or real-time information
- Use pii_protection_tool only when explicitly combining with database queries that need masking
- You can select multiple tools if the query requires information from different sources

IMPORTANT: Return ONLY the tool indices as comma-separated numbers (e.g., "0,3" or "5").
Do not include any explanations or other text.

Selected tool indices:"""

        # 3. Parse LLM response to get tool indices
        try:
            response = self.llm.complete(prompt)
            response_text = str(response).strip()

            # Parse the tool indices
            selected_indices = []
            for part in response_text.split(','):
                part = part.strip()
                # Extract first number found
                for char in part:
                    if char.isdigit():
                        try:
                            idx = int(part)
                            if 0 <= idx < len(all_tools):
                                selected_indices.append(idx)
                            break
                        except ValueError:
                            continue
                        break

            if not selected_indices:
                # Default to first tool if parsing fails
                selected_indices = [0]

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Routing failed: {e}, using default tool")
            selected_indices = [0]

        # 5. Execute selected tools and collect results
        results = []
        for idx in selected_indices:
            if idx >= len(all_tools):
                continue

            tool = all_tools[idx]
            tool_name = tool.metadata.name if hasattr(tool, 'metadata') else f"tool_{idx}"
            tool_desc = tool.metadata.description if hasattr(tool, 'metadata') else "No description"

            try:
                # Execute the tool
                result = tool.call(query)

                # Extract content from ToolOutput if needed
                if hasattr(result, 'content'):
                    result_str = str(result.content)
                else:
                    result_str = str(result)

                # 6. Apply PII protection to database results
                result_str = self._check_and_apply_pii_protection(tool_name, result_str)

                results.append((tool_name, tool_desc, result_str))

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Tool {tool_name} failed: {e}")
                results.append((tool_name, tool_desc, f"Error: {e}"))

        return results
    
    def query(self, question: str, verbose: bool = None) -> str:
        """Process query with dynamic tool routing and result synthesis
        
        This is the main entry point for the financial agent. It handles:
        1. Tool routing and selection using LLM
        2. Multi-tool execution 
        3. Result synthesis for comprehensive answers
        4. Automatic PII protection
        
        Args:
            question: User's financial question
            verbose: Whether to show detailed processing info
            
        Returns:
            Comprehensive answer synthesized from relevant tools
        """
        
        # Use instance verbose if parameter not provided
        if verbose is None:
            verbose = self.verbose
        
        # Ensure tools are initialized
        if not self._tools_initialized:
            self.setup()
            self._tools_initialized = True
        
        if verbose:
            print(f"🎯 Query: {question}")

        # 1. Route query to appropriate tools using _route_query()
        tool_results = self._route_query(question)

        if not tool_results:
            return "No tools were able to process the query."

        # 2. Display tool selection info if verbose
        if verbose:
            print(f"🔧 Selected {len(tool_results)} tool(s):")
            for tool_name, tool_desc, _ in tool_results:
                print(f"   - {tool_name}")

        # 3. If single tool result, return it directly
        if len(tool_results) == 1:
            tool_name, tool_desc, result = tool_results[0]
            if verbose:
                print(f"✅ Answer from {tool_name}")
            return result

        # 4. If multiple tool results, synthesize using LLM
        if verbose:
            print("🔄 Synthesizing results from multiple tools...")

        # Build synthesis prompt
        results_text = ""
        for i, (tool_name, tool_desc, result) in enumerate(tool_results, 1):
            results_text += f"\n\n--- Source {i}: {tool_name} ---\n{result}\n"

        synthesis_prompt = f"""You are a financial analyst synthesizing information from multiple sources.
Combine the following information to provide a comprehensive answer to the user's question.

User Question: {question}

Information from Multiple Sources:{results_text}

Instructions:
- Synthesize the information into a coherent, comprehensive answer
- Cite which sources provide which information
- Resolve any conflicts between sources
- Maintain accuracy and don't add information not present in the sources
- If PII protection was applied, acknowledge this appropriately

Comprehensive Answer:"""

        try:
            # 5. Return comprehensive answer
            synthesis_response = self.llm.complete(synthesis_prompt)
            synthesized_answer = str(synthesis_response).strip()

            if verbose:
                print("✅ Synthesis complete")

            return synthesized_answer

        except Exception as e:
            if verbose:
                print(f"⚠️  Synthesis failed: {e}, returning individual results")

            # Fallback: return concatenated results
            answer = f"Answer (from {len(tool_results)} sources):\n\n"
            for tool_name, tool_desc, result in tool_results:
                answer += f"\n--- From {tool_name} ---\n{result}\n"

            return answer
    
    def get_available_tools(self) -> Dict[str, Any]:
        """
        Get information about available tools with full compatibility.
        
        Returns:
            Dictionary with comprehensive tool information
        """
        return {
            "document_tools": ["apple", "google", "tesla"] if len(self.document_tools) >= 3 else [],
            "function_tools": ["sql", "market", "pii"] if len(self.function_tools) >= 3 else [],
            "total_tools": len(self.document_tools) + len(self.function_tools),
            "document_tool_count": len(self.document_tools),
            "function_tool_count": len(self.function_tools)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status with full compatibility.
        
        Returns:
            Dictionary with detailed status information
        """
        tool_count = len(self.document_tools) + len(self.function_tools)
        system_ready = len(self.document_tools) >= 3 and len(self.function_tools) >= 3
        
        return {
            "companies": self.companies,
            "document_tools": len(self.document_tools),
            "function_tools": len(self.function_tools),
            "total_tools": tool_count,
            "ready": system_ready,
            "architecture": "modular",
            "coordinator_ready": system_ready,
            "available_companies": ['AAPL', 'GOOGL', 'TSLA'],
            "capabilities": [
                "Document analysis (10-K filings)",
                "Database queries (customer portfolios)",
                "Real-time market data",
                "PII protection",
                "Multi-tool coordination",
                "Intelligent routing"
            ],
            "system_ready": system_ready
        }
