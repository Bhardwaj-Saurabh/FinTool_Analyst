"""
Function Tools Module - Database queries, market data, and PII protection

This module provides function-based tools for SQL generation, market data retrieval,
and PII protection. These are the core business logic tools that enable the agent
to access database information and current market data.

Learning Objectives:
- Understand function tool creation with LlamaIndex
- Implement database querying with SQL generation
- Create market data retrieval tools
- Build PII protection mechanisms
- Learn about real-time API integration

Your Task: Complete the missing implementations marked with YOUR CODE HERE

Key Concepts:
1. FunctionTool Creation: Wrap Python functions as LlamaIndex tools
2. SQL Generation: Use LLM to generate SQL from natural language
3. Database Operations: Execute SQL queries and format results  
4. API Integration: Fetch real-time market data from external APIs
5. PII Protection: Automatically mask sensitive information
"""

import logging
import sqlite3
import random
from pathlib import Path
from typing import List, Tuple

# LlamaIndex imports
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class FunctionToolsManager:
    """Manager for all function tools - Database, market data, and PII protection"""
    
    def __init__(self, verbose: bool = False):
        """Initialize function tools manager
        
        Args:
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.db_path = self.project_root / "data" / "financial.db"
        
        # Database schema for SQL generation
        self.db_schema = self._get_database_schema()
        
        # Storage for tools
        self.function_tools = []
        
        self._configure_settings()
        
        if self.verbose:
            print("✅ Function Tools Manager Initialized")
    
    def _configure_settings(self):
        """Configure LlamaIndex settings

        TODO: Set up the LLM for SQL generation and other AI tasks

        Requirements:
        - Create OpenAI LLM with "gpt-3.5-turbo" model and temperature=0
        - Set Settings.llm and Settings.embed_model
        - Store LLM reference in self.llm for use in tools

        IMPORTANT NOTE FOR VOCAREUM:
        LlamaIndex requires the api_base parameter to work with Vocareum's OpenAI endpoint.
        Get the base URL from environment: os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")
        Pass it as api_base parameter to both OpenAI() and OpenAIEmbedding() constructors.

        Hint: This is similar to document_tools configuration
        """
        import os

        # Get the Vocareum API base URL from environment
        api_base = os.getenv("OPENAI_API_BASE", "https://openai.vocareum.com/v1")

        # Configure OpenAI LLM with gpt-3.5-turbo and temperature=0
        self.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_base=api_base
        )

        # Configure OpenAI embeddings
        embed_model = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_base=api_base
        )

        # Set global Settings
        Settings.llm = self.llm
        Settings.embed_model = embed_model

        if self.verbose:
            print("✅ LlamaIndex settings configured with OpenAI models")
    
    def _get_database_schema(self) -> str:
        """Get enhanced database schema with relationships for SQL generation
        
        This method reads the database structure and returns a comprehensive
        schema description that helps the LLM generate better SQL queries.
        
        Returns:
            String containing detailed database schema with table relationships
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get table names to verify database connection
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Return comprehensive schema for SQL generation
            schema_info = """Enhanced Database Schema with Relationships:

TABLE: customers (Customer Information)
- id (PRIMARY KEY, INTEGER) - Unique customer identifier
- first_name (TEXT) - Customer first name
- last_name (TEXT) - Customer last name  
- email (TEXT) - Customer email address
- phone (TEXT) - Customer phone number
- investment_profile (TEXT) - conservative/moderate/aggressive
- risk_tolerance (TEXT) - low/medium/high

TABLE: portfolio_holdings (Customer Stock Holdings)
- id (PRIMARY KEY, INTEGER) - Unique holding record
- customer_id (FOREIGN KEY → customers.id) - Links to customer
- symbol (TEXT) - Stock symbol like 'AAPL', 'TSLA', 'MSFT', 'GOOGL'
- shares (REAL) - Number of shares owned
- purchase_price (REAL) - Price when purchased
- current_value (REAL) - Current total value of holding

TABLE: companies (Company Master Data)
- id (PRIMARY KEY, INTEGER) - Unique company identifier
- symbol (TEXT) - Stock symbol like 'AAPL', 'TSLA', 'MSFT', 'GOOGL'
- name (TEXT) - Company name like 'Apple Inc', 'Tesla Inc'
- sector (TEXT) - Business sector (technology, automotive, etc.)
- market_cap (REAL) - Market capitalization

TABLE: financial_metrics (Company Financial Data)
- id (PRIMARY KEY, INTEGER) - Unique metrics record
- symbol (FOREIGN KEY → companies.symbol) - Stock symbol
- revenue (REAL) - Company revenue
- net_income (REAL) - Net income
- eps (REAL) - Earnings per share
- pe_ratio (REAL) - Price to earnings ratio
- debt_to_equity (REAL) - Debt to equity ratio
- roe (REAL) - Return on equity

TABLE: market_data (Current Market Information)
- id (PRIMARY KEY, INTEGER) - Unique market record
- symbol (FOREIGN KEY → companies.symbol) - Stock symbol
- close_price (REAL) - Latest closing price
- volume (INTEGER) - Trading volume
- market_cap (REAL) - Current market cap
- date (TEXT) - Date of data

COMMON QUERY PATTERNS & JOINS:

1. Customer holdings with names:
   SELECT c.first_name, c.last_name, ph.symbol, ph.shares, ph.current_value
   FROM customers c 
   JOIN portfolio_holdings ph ON c.id = ph.customer_id

2. Holdings with company information:
   SELECT ph.symbol, co.name, ph.shares, ph.current_value, co.sector
   FROM portfolio_holdings ph
   JOIN companies co ON ph.symbol = co.symbol

3. Holdings with current market prices:
   SELECT ph.symbol, ph.shares, ph.current_value, md.close_price
   FROM portfolio_holdings ph
   JOIN market_data md ON ph.symbol = md.symbol

4. Complete customer portfolio view:
   SELECT c.first_name, c.last_name, co.name, ph.shares, 
          ph.current_value, md.close_price, co.sector
   FROM customers c
   JOIN portfolio_holdings ph ON c.id = ph.customer_id
   JOIN companies co ON ph.symbol = co.symbol
   JOIN market_data md ON ph.symbol = md.symbol

KEY TIPS:
- Use LIKE '%Tesla%' or LIKE '%Apple%' for company name searches
- Use symbol = 'TSLA', 'AAPL', 'MSFT', 'GOOGL' for exact stock matches
- JOIN portfolio_holdings with customers to get customer names
- JOIN with companies to get full company names and sectors
- JOIN with market_data to get current prices and volumes
"""
            
            conn.close()
            return schema_info
            
        except Exception as e:
            return f"Schema error: {e}\n\nFallback basic schema available."
    
    def create_function_tools(self):
        """Create function tools for database, market data, and PII protection
        
        This method creates three main function tools:
        1. Database Query Tool - Generates and executes SQL queries
        2. Market Search Tool - Fetches real-time stock data
        3. PII Protection Tool - Masks sensitive information
        
        Returns:
            List of FunctionTool objects
        """
        if self.verbose:
            print("🛠️ Creating function tools...")
        
        # Clear existing tools
        self.function_tools = []
        
        # TODO: Create the three main function tools
        # Implement these three nested functions and wrap them with FunctionTool:
        # 1. database_query_tool - Natural language to SQL conversion and execution
        # 2. finance_market_search_tool - Real-time Yahoo Finance API integration
        # 3. pii_protection_tool - Automatic PII detection and masking
        
        # 1. DATABASE QUERY TOOL
        def database_query_tool(query: str) -> str:
            """Generate and execute SQL queries for customer/portfolio database
            
            This tool takes a natural language query, converts it to SQL using
            the LLM, executes it against the database, and returns formatted results.
            
            Args:
                query: Natural language question about the database
                
            Returns:
                String containing SQL query and formatted results
            """
            
            def generate_sql(query_text: str, error_context: str = None) -> str:
                """Generate SQL query from natural language using LLM"""
                # Build prompt that includes database schema and query
                prompt = f"""You are a SQL expert. Generate a SQLite query to answer the question.

Database Schema:
{self.db_schema}

Question: {query_text}
"""
                # Add error context if this is a retry
                if error_context:
                    prompt += f"\n\nPrevious attempt failed with error: {error_context}\nPlease fix the SQL query."

                prompt += """

IMPORTANT RULES:
- Return ONLY the SQL query, no explanations
- Use proper SQLite syntax
- Do not use markdown code blocks
- Return a single SQL statement only
- Use JOIN clauses properly for related data

SQL Query:"""

                # Use self.llm.complete() to generate SQL
                response = self.llm.complete(prompt)
                sql = str(response).strip()

                # Clean up response (remove markdown, handle multiple statements)
                # Remove markdown code blocks if present
                if "```sql" in sql:
                    sql = sql.split("```sql")[1].split("```")[0].strip()
                elif "```" in sql:
                    sql = sql.split("```")[1].split("```")[0].strip()

                # Remove any text after semicolon (multiple statements)
                if ";" in sql:
                    sql = sql.split(";")[0].strip() + ";"

                return sql
            
            def execute_sql(sql_query: str) -> Tuple[bool, list, list, str]:
                """Execute SQL and return (success, results, column_names, error)"""
                try:
                    # Connect to database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()

                    # Execute query
                    cursor.execute(sql_query)

                    # Extract results and column names
                    results = cursor.fetchall()
                    column_names = [description[0] for description in cursor.description] if cursor.description else []

                    conn.close()

                    # Return success with results and column names
                    return True, results, column_names, None

                except Exception as e:
                    # Return failure with error message
                    return False, None, None, str(e)
            
            try:
                # 1. Generate SQL from natural language query
                sql_query = generate_sql(query)

                # 2. Execute the SQL and get results
                success, results, column_names, error = execute_sql(sql_query)

                # 3. If execution fails, retry with error context
                if not success and error:
                    # Retry with error context
                    sql_query = generate_sql(query, error_context=error)
                    success, results, column_names, error = execute_sql(sql_query)

                    # If still fails, return error
                    if not success:
                        return f"SQL Query: {sql_query}\n\nExecution Error: {error}"

                # 4. Format results with column names
                if not results:
                    return f"SQL Query: {sql_query}\n\nNo results found."

                # Format results as a table
                result_str = f"SQL Query: {sql_query}\n\n"
                result_str += "Results:\n"
                result_str += " | ".join(column_names) + "\n"
                result_str += "-" * (len(" | ".join(column_names))) + "\n"

                for row in results:
                    result_str += " | ".join(str(val) for val in row) + "\n"

                return result_str

            except Exception as e:
                return f"Database system error: {e}"
        
        # 2. MARKET DATA TOOL
        def finance_market_search_tool(query: str) -> str:
            """Get real current stock prices and market information
            
            This tool fetches real-time stock data from Yahoo Finance API
            for Apple (AAPL), Tesla (TSLA), and Google (GOOGL).
            
            Args:
                query: Natural language query mentioning companies
                
            Returns:
                String containing current market data
            """
            
            def get_real_stock_data(symbol: str) -> dict:
                """Fetch real stock data from Yahoo Finance API"""
                try:
                    import requests

                    # Make API call to Yahoo Finance
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                    response = requests.get(url, timeout=10)

                    if response.status_code != 200:
                        return {'success': False, 'error': f'API returned status {response.status_code}'}

                    data = response.json()

                    # Extract data from response
                    if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                        return {'success': False, 'error': 'Invalid API response format'}

                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})

                    # Extract: current price, previous close, volume, market cap
                    current_price = meta.get('regularMarketPrice', 0)
                    previous_close = meta.get('previousClose', 0)
                    volume = meta.get('regularMarketVolume', 0)
                    market_cap = meta.get('marketCap', 0)

                    # Calculate: price change and change percentage
                    price_change = current_price - previous_close
                    change_percent = (price_change / previous_close * 100) if previous_close != 0 else 0

                    # Return: Dictionary with stock data and success flag
                    return {
                        'success': True,
                        'symbol': symbol,
                        'current_price': current_price,
                        'previous_close': previous_close,
                        'price_change': price_change,
                        'change_percent': change_percent,
                        'volume': volume,
                        'market_cap': market_cap
                    }

                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            try:
                # Identify companies mentioned in the query
                # Map company names/symbols to ticker symbols
                query_lower = query.lower()
                companies_to_fetch = []

                company_mapping = {
                    'apple': 'AAPL',
                    'aapl': 'AAPL',
                    'tesla': 'TSLA',
                    'tsla': 'TSLA',
                    'google': 'GOOGL',
                    'googl': 'GOOGL',
                    'alphabet': 'GOOGL'
                }

                # Check which companies are mentioned
                for name, symbol in company_mapping.items():
                    if name in query_lower:
                        if symbol not in companies_to_fetch:
                            companies_to_fetch.append(symbol)

                # If no specific company mentioned, return all three
                if not companies_to_fetch:
                    companies_to_fetch = ['AAPL', 'TSLA', 'GOOGL']

                # Fetch stock data for each identified company
                results = []
                for symbol in companies_to_fetch:
                    stock_data = get_real_stock_data(symbol)

                    if stock_data['success']:
                        results.append(stock_data)
                    else:
                        # Handle API failures with appropriate fallbacks
                        results.append({
                            'symbol': symbol,
                            'error': stock_data.get('error', 'Unknown error')
                        })

                # Format results with price, change, volume
                if not results:
                    return "No market data available for the requested companies."

                output = "Current Market Data:\n\n"
                for data in results:
                    if 'error' in data:
                        output += f"{data['symbol']}: Error fetching data - {data['error']}\n"
                    else:
                        change_symbol = "+" if data['price_change'] >= 0 else ""
                        output += f"{data['symbol']}:\n"
                        output += f"  Current Price: ${data['current_price']:.2f}\n"
                        output += f"  Change: {change_symbol}${data['price_change']:.2f} ({change_symbol}{data['change_percent']:.2f}%)\n"
                        output += f"  Volume: {data['volume']:,}\n"
                        if data['market_cap']:
                            output += f"  Market Cap: ${data['market_cap']:,.0f}\n"
                        output += "\n"

                return output.strip()

            except Exception as e:
                return f"Market data error: {e}"
        
        # 3. PII PROTECTION TOOL
        def pii_protection_tool(database_results: str, column_names: str) -> str:
            """Automatically mask PII fields in database results
            
            This tool identifies and masks personally identifiable information
            in database query results based on column names and content patterns.
            
            Args:
                database_results: Raw database results as string
                column_names: List of column names (as string)
                
            Returns:
                String with PII fields masked for privacy protection
            """
            
            def detect_pii_fields(field_names: list) -> set:
                """Detect which fields contain PII based on field names"""
                # Create patterns for common PII field names
                pii_patterns = [
                    'email', 'phone', 'first_name', 'last_name', 'name',
                    'address', 'ssn', 'social_security', 'birth_date', 'dob',
                    'passport', 'license', 'credit_card', 'account_number'
                ]

                detected_pii = set()

                # Check field names against patterns
                for field in field_names:
                    field_lower = field.lower()
                    for pattern in pii_patterns:
                        if pattern in field_lower:
                            detected_pii.add(field)
                            break

                return detected_pii
            
            def mask_field_value(field_name: str, value: str) -> str:
                """Apply appropriate masking based on field type"""
                if value is None:
                    return str(value)

                value_str = str(value)
                field_lower = field_name.lower()

                # Email masking: abc@gmail.com -> ***@gmail.com
                if 'email' in field_lower:
                    if '@' in value_str:
                        return '***@' + value_str.split('@')[1]
                    return '***'

                # Phone masking: 123-456-7890 -> ***-***-7890
                if 'phone' in field_lower:
                    if '-' in value_str:
                        parts = value_str.split('-')
                        return '-'.join(['***'] * (len(parts) - 1) + [parts[-1]])
                    return '***'

                # Name masking: John -> ****
                if 'name' in field_lower:
                    return '****'

                # Default masking for other PII fields
                return '****'
            
            try:
                # Parse column names from string format
                import ast
                try:
                    # Try to parse as Python list string
                    column_list = ast.literal_eval(column_names)
                except:
                    # If that fails, try comma-separated or other format
                    if isinstance(column_names, str):
                        column_list = [col.strip() for col in column_names.replace('[', '').replace(']', '').replace("'", "").split(',')]
                    else:
                        column_list = column_names

                # Detect PII fields
                pii_fields = detect_pii_fields(column_list)

                if not pii_fields:
                    # No PII detected, return original results
                    return database_results

                # Parse database results line by line and apply masking
                lines = database_results.split('\n')
                masked_lines = []

                for line in lines:
                    # Skip empty lines or separator lines
                    if not line.strip() or line.startswith('-') or line.startswith('='):
                        masked_lines.append(line)
                        continue

                    # Check if this is a data line (contains pipe separators)
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]

                        # Apply masking to PII columns
                        masked_parts = []
                        for i, part in enumerate(parts):
                            if i < len(column_list) and column_list[i] in pii_fields:
                                masked_parts.append(mask_field_value(column_list[i], part))
                            else:
                                masked_parts.append(part)

                        masked_lines.append(' | '.join(masked_parts))
                    else:
                        masked_lines.append(line)

                # Add notice about which fields were masked
                result = '\n'.join(masked_lines)
                if pii_fields:
                    result += f"\n\n[PII Protection Applied: {', '.join(sorted(pii_fields))} fields masked]"

                return result

            except Exception as e:
                # If parsing fails, return original with warning
                return f"{database_results}\n\n[PII Protection Warning: Could not parse results - {e}]"
        
        # Create FunctionTool objects for each function
        # 1. Database Query Tool
        db_tool = FunctionTool.from_defaults(
            fn=database_query_tool,
            name="database_query_tool",
            description=(
                "Query the customer portfolio database using natural language. "
                "This tool generates SQL queries from natural language questions and returns "
                "customer information, portfolio holdings, investment positions, and financial data. "
                "Use this to find customer details, stock holdings, investment profiles, and account information."
            )
        )
        self.function_tools.append(db_tool)

        # 2. Market Search Tool
        market_tool = FunctionTool.from_defaults(
            fn=finance_market_search_tool,
            name="finance_market_search_tool",
            description=(
                "Get real-time current stock market data and prices for Apple (AAPL), "
                "Tesla (TSLA), and Google (GOOGL). Returns current price, price changes, "
                "trading volume, and market capitalization. Use this for live market information "
                "and current stock performance data."
            )
        )
        self.function_tools.append(market_tool)

        # 3. PII Protection Tool
        pii_tool = FunctionTool.from_defaults(
            fn=pii_protection_tool,
            name="pii_protection_tool",
            description=(
                "Automatically detect and mask personally identifiable information (PII) "
                "in database results. Protects sensitive data like names, emails, phone numbers, "
                "and addresses by masking them before display. Use this tool to process database "
                "results that may contain customer personal information."
            )
        )
        self.function_tools.append(pii_tool)

        if self.verbose:
            print("   ✅ Function tools created")

        return self.function_tools
    
    def get_tools(self):
        """Get all function tools
        
        Returns:
            List of FunctionTool objects
        """
        return self.function_tools

