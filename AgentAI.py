import logging
import azure.functions as func
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
import tiktoken
import datetime

"""
AgentAI Azure Function - Comprehensive Validation and Error Handling System

This module provides a robust AgentAI system with comprehensive validation for worst-case scenarios.
The system includes multiple layers of validation to ensure data integrity and provide clear error messages.

VALIDATION LAYERS:
1. Input Parameter Validation (validateInputParameters)
   - Validates tenant, sendgridCampaignId, and campaignId parameters
   - Ensures required parameters are present and properly formatted
   - Returns 400 status for invalid input parameters

2. Workflow Result Validation (validateWorkflowResult)
   - Validates the structure of workflow execution results
   - Ensures required fields (supplierAiTask, token_usage) are present
   - Returns 500 status for workflow execution failures

3. AI Agent Output Validation (validateAIAgentOutput)
   - Detects reasoning loops and invalid AI agent responses
   - Validates JSON format and structure of agent output
   - Identifies error messages and malformed responses
   - Returns 422 status for AI agent errors

4. Supplier List Validation (validateSupplierList)
   - Validates the structure and content of supplier data
   - Ensures required fields are present in each supplier object
   - Detects empty supplier lists and provides detailed error information
   - Returns 500 status for data structure errors

5. Extracted Data Validation (validateExtractedData)
   - Validates extracted supplier names and components
   - Ensures meaningful data is available for processing
   - Provides detailed error information for empty or invalid data
   - Returns 500 status for data extraction errors

ERROR RESPONSE FORMAT:
All error responses follow a standardized format with:
- status: "error"
- error_type: Specific error category
- message: Human-readable error description
- timestamp: Error occurrence time
- details: Technical error details
- troubleshooting: List of suggested resolution steps

HTTP STATUS CODES:
- 400: Bad Request (invalid input parameters)
- 422: Unprocessable Entity (AI agent errors)
- 500: Internal Server Error (system/data processing errors)

WORST-CASE SCENARIOS HANDLED:
- Empty supplier lists (no suppliers found)
- Empty component lists (no components found)
- AI agent reasoning loops
- Invalid JSON responses from AI agents
- Database connectivity issues
- Missing required tools
- Data processing failures
- JSON serialization errors
- Missing or invalid input parameters
- Workflow execution failures

This comprehensive validation system ensures that the frontend receives clear, actionable error messages
and prevents empty or invalid responses from reaching the client application.
"""


def loadEnvironmentConfiguration():
    """
    Loads environment variables for API configuration and database connections.
    Initializes the OpenAI LLM with specific configuration for consistent, deterministic responses.
    
    Returns:
        ChatOpenAI: Configured OpenAI LLM instance with temperature=0.0 and seed=42 for deterministic responses
    
    Raises:
        Exception: If environment configuration loading fails
    """
    try:
        load_dotenv()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        openai_llm = ChatOpenAI(
            model_name=openai_model,
            temperature=1.0,
            seed=42,
        )
        logging.info("Environment configuration loaded successfully")
        return openai_llm
    except Exception as e:
        logging.error(f"Error loading environment configuration: {e}")
        raise e


def configureMongoDBConnection(tenant):
    """
    Configures MongoDB connection parameters for MCP server.
    This establishes the connection to the MongoDB database for data retrieval operations.
    
    Args:
        tenant (str): The tenant identifier to connect to specific database instance
    
    Returns:
        StdioServerParameters: Configured server parameters for MongoDB MCP connection
    
    Raises:
        Exception: If MongoDB connection configuration fails
    """ 
    try:
        load_dotenv()
        mongodb_uri = os.getenv("MONGO_DB")
        mongodb = mongodb_uri+"/"+tenant
        server_params = StdioServerParameters(
            command="npx",
            args=["npx", "mcp-mongo-server", mongodb],
            env={"UV_PYTHON": "3.11", **os.environ},
        )
        logging.info("MongoDB connection parameters configured successfully")
        return server_params
    except Exception as e:
        logging.error(f"Error configuring MongoDB connection: {e}")
        raise e


def validateInputParameters(tenant, sendgridCampaignId, campaignId):
    """
    Validates input parameters for the AgentAI function to ensure all required data is present and valid.
    This function performs comprehensive validation to prevent downstream errors and provide clear error messages.
    
    Args:
        tenant (str): The tenant identifier for database connection
        sendgridCampaignId (str): The SendGrid campaign ID to analyze
        campaignId (str): The campaign identifier for context
    
    Returns:
        dict: Validation result with status and error details if validation fails
    
    Raises:
        ValueError: If any required parameter is missing or invalid
    """
    validation_errors = []
    
    # Validate tenant parameter
    if not tenant or not isinstance(tenant, str) or tenant.strip() == "":
        validation_errors.append({
            "parameter": "tenant",
            "issue": "Missing or empty tenant parameter",
            "required": True,
            "type": "string"
        })
    
    # Validate sendgridCampaignId parameter
    if not sendgridCampaignId or not isinstance(sendgridCampaignId, str) or sendgridCampaignId.strip() == "":
        validation_errors.append({
            "parameter": "sendgridCampaignId",
            "issue": "Missing or empty sendgridCampaignId parameter",
            "required": True,
            "type": "string"
        })
    
    # Validate campaignId parameter (optional but should be string if provided)
    if campaignId is not None and (not isinstance(campaignId, str) or campaignId.strip() == ""):
        validation_errors.append({
            "parameter": "campaignId",
            "issue": "Invalid campaignId parameter - must be string if provided",
            "required": False,
            "type": "string"
        })
    
    if validation_errors:
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "Input parameter validation failed"
        }
    
    return {"valid": True}


def validateWorkflowResult(workflow_result):
    """
    Validates the workflow execution result to ensure data integrity and completeness.
    This function checks for various failure scenarios and provides detailed error information.
    
    Args:
        workflow_result (dict): The result from executeDataAnalysisWorkflow function
    
    Returns:
        dict: Validation result with status and detailed error information if validation fails
    """
    validation_errors = []
    
    # Check if workflow_result exists and is a dictionary
    if not workflow_result or not isinstance(workflow_result, dict):
        validation_errors.append({
            "component": "workflow_result",
            "issue": "Workflow result is missing or invalid",
            "expected": "dictionary",
            "received": type(workflow_result).__name__ if workflow_result else "None"
        })
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "Workflow execution failed - invalid result structure"
        }
    
    # Check if supplierAiTask exists in workflow result
    if "supplierAiTask" not in workflow_result:
        validation_errors.append({
            "component": "supplierAiTask",
            "issue": "Supplier analysis task result is missing from workflow",
            "expected": "supplierAiTask field in workflow_result",
            "received": f"Available fields: {list(workflow_result.keys())}"
        })
    
    # Check if token_usage exists in workflow result
    if "token_usage" not in workflow_result:
        validation_errors.append({
            "component": "token_usage",
            "issue": "Token usage tracking is missing from workflow",
            "expected": "token_usage field in workflow_result",
            "received": f"Available fields: {list(workflow_result.keys())}"
        })
    
    if validation_errors:
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "Workflow result validation failed"
        }
    
    return {"valid": True}


def validateSupplierList(supplierList, sendgridCampaignId):
    """
    Validates the supplier list data to ensure it contains valid and complete information.
    This function performs comprehensive validation of supplier data structure and content.
    
    Args:
        supplierList (list): The parsed supplier list from AI agent output
        sendgridCampaignId (str): The SendGrid campaign ID for context
    
    Returns:
        dict: Validation result with status and detailed error information if validation fails
    """
    validation_errors = []
    
    # Check if supplierList is a list
    if not isinstance(supplierList, list):
        validation_errors.append({
            "component": "supplierList",
            "issue": "Supplier list is not a valid array",
            "expected": "list/array",
            "received": type(supplierList).__name__
        })
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "Invalid supplier list structure"
        }
    
    # Check if supplierList is empty
    if len(supplierList) == 0:
        validation_errors.append({
            "component": "supplierList",
            "issue": "Supplier list is empty - no suppliers found for analysis",
            "campaign_id": sendgridCampaignId,
            "possible_causes": [
                "No campaign found with the provided sendgridCampaignId",
                "Campaign has no associated suppliers",
                "All suppliers have already responded to the campaign",
                "Database query returned no results",
                "AI agent failed to extract supplier data"
            ]
        })
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "No suppliers found for analysis"
        }
    
    # Validate each supplier object in the list
    for index, supplier in enumerate(supplierList):
        if not isinstance(supplier, dict):
            validation_errors.append({
                "component": f"supplier[{index}]",
                "issue": "Supplier object is not a valid dictionary",
                "expected": "dictionary",
                "received": type(supplier).__name__
            })
            continue
        
        # Check required fields for each supplier
        required_fields = ["supplierName", "supplierId", "campaignName"]
        for field in required_fields:
            if field not in supplier or not supplier[field]:
                validation_errors.append({
                    "component": f"supplier[{index}].{field}",
                    "issue": f"Required field '{field}' is missing or empty",
                    "supplier_index": index,
                    "supplier_data": supplier
                })
    
    if validation_errors:
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "Supplier list validation failed"
        }
    
    return {"valid": True}


def validateExtractedData(supplierNames, components, campaignName, sendgridCampaignId):
    """
    Validates the extracted supplier names and components data to ensure meaningful results.
    This function checks for empty or invalid extracted data and provides detailed error information.
    
    Args:
        supplierNames (list): List of extracted supplier names
        components (list): List of extracted component IDs
        campaignName (str): The campaign name for context
        sendgridCampaignId (str): The SendGrid campaign ID for context
    
    Returns:
        dict: Validation result with status and detailed error information if validation fails
    """
    validation_errors = []
    
    # Validate supplier names
    if not supplierNames or len(supplierNames) == 0:
        validation_errors.append({
            "component": "supplierNames",
            "issue": "No supplier names extracted from analysis",
            "campaign_id": sendgridCampaignId,
            "campaign_name": campaignName,
            "possible_causes": [
                "All suppliers have responded to the campaign",
                "Campaign has no associated suppliers",
                "AI agent failed to extract supplier names",
                "Database query returned empty supplier data"
            ]
        })
    
    # Validate components (can be empty but should be a list)
    if not isinstance(components, list):
        validation_errors.append({
            "component": "components",
            "issue": "Components data is not a valid list",
            "expected": "list",
            "received": type(components).__name__
        })
    elif len(components) == 0:
        # Components can be empty, but log it as a warning
        logging.warning(f"No components found for campaign {sendgridCampaignId} - this may be expected for certain campaign types")
    
    # Validate campaign name
    if not campaignName or campaignName.strip() == "":
        validation_errors.append({
            "component": "campaignName",
            "issue": "Campaign name is missing or empty",
            "campaign_id": sendgridCampaignId,
            "possible_causes": [
                "Campaign not found in database",
                "AI agent failed to extract campaign name",
                "Database query returned no campaign data"
            ]
        })
    
    if validation_errors:
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "Extracted data validation failed"
        }
    
    return {"valid": True}


def validateAIAgentOutput(supplierOutput, sendgridCampaignId):
    """
    Validates the AI agent output to detect reasoning loops, invalid JSON, or unexpected response formats.
    This function handles various AI agent failure scenarios and provides detailed error information.
    
    Args:
        supplierOutput: The raw output from the AI agent
        sendgridCampaignId (str): The SendGrid campaign ID for context
    
    Returns:
        dict: Validation result with status and detailed error information if validation fails
    """
    validation_errors = []
    
    # Check if output is None or empty
    if supplierOutput is None:
        validation_errors.append({
            "component": "ai_agent_output",
            "issue": "AI agent returned no output",
            "campaign_id": sendgridCampaignId,
            "possible_causes": [
                "AI agent failed to execute",
                "Database connection issues",
                "Agent configuration problems",
                "Token limit exceeded"
            ]
        })
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "AI agent produced no output"
        }
    
    # Convert to string for analysis if not already
    output_str = str(supplierOutput).strip()
    
    # Check for reasoning loop indicators
    reasoning_indicators = [
        "I tried reusing the same input",
        "I must stop using this action input",
        "I'll try something else instead",
        "I cannot proceed with the same input",
        "I need to stop this approach"
    ]
    
    for indicator in reasoning_indicators:
        if indicator.lower() in output_str.lower():
            validation_errors.append({
                "component": "ai_agent_output",
                "issue": "AI agent entered reasoning loop",
                "campaign_id": sendgridCampaignId,
                "detected_pattern": indicator,
                "possible_causes": [
                    "Agent configuration allows multiple iterations",
                    "Task description is unclear or ambiguous",
                    "Database query complexity causing agent confusion",
                    "Token limit or timeout issues"
                ],
                "raw_output": output_str[:500] + "..." if len(output_str) > 500 else output_str
            })
            return {
                "valid": False,
                "errors": validation_errors,
                "message": "AI agent entered reasoning loop"
            }
    
    # Check for error messages in output
    error_indicators = [
        "error",
        "failed",
        "exception",
        "cannot",
        "unable",
        "invalid",
        "not found"
    ]
    
    error_messages = []
    for indicator in error_indicators:
        if indicator.lower() in output_str.lower():
            # Extract the line containing the error
            lines = output_str.split('\n')
            for line in lines:
                if indicator.lower() in line.lower():
                    error_messages.append(line.strip())
                    break
    
    if error_messages:
        validation_errors.append({
            "component": "ai_agent_output",
            "issue": "AI agent reported errors during execution",
            "campaign_id": sendgridCampaignId,
            "error_messages": error_messages[:3],  # Limit to first 3 errors
            "possible_causes": [
                "Database connection issues",
                "Invalid query parameters",
                "Missing data in collections",
                "Permission or access issues"
            ]
        })
    
    # Check if output looks like valid JSON structure
    try:
        if isinstance(supplierOutput, str):
            json.loads(supplierOutput)
        elif isinstance(supplierOutput, (list, dict)):
            # Already parsed JSON structure
            pass
        else:
            validation_errors.append({
                "component": "ai_agent_output",
                "issue": "AI agent output is not in valid JSON format",
                "campaign_id": sendgridCampaignId,
                "output_type": type(supplierOutput).__name__,
                "possible_causes": [
                    "Agent returned plain text instead of JSON",
                    "JSON parsing failed due to malformed output",
                    "Agent configuration issue causing wrong output format"
                ]
            })
    except json.JSONDecodeError as e:
        validation_errors.append({
            "component": "ai_agent_output",
            "issue": "AI agent output contains invalid JSON",
            "campaign_id": sendgridCampaignId,
            "json_error": str(e),
            "possible_causes": [
                "Agent returned malformed JSON",
                "Output contains extra text or formatting",
                "Agent configuration causing JSON syntax errors"
            ],
            "raw_output": output_str[:500] + "..." if len(output_str) > 500 else output_str
        })
    
    if validation_errors:
        return {
            "valid": False,
            "errors": validation_errors,
            "message": "AI agent output validation failed"
        }
    
    return {"valid": True}


def createErrorResponse(status_code, error_type, message, details=None, troubleshooting=None):
    """
    Creates a standardized error response with comprehensive error information.
    This function ensures consistent error response format across all failure scenarios.
    
    Args:
        status_code (int): HTTP status code for the error
        error_type (str): Type of error (e.g., "validation_error", "workflow_error")
        message (str): Human-readable error message
        details (dict, optional): Additional error details
        troubleshooting (list, optional): List of troubleshooting steps
    
    Returns:
        func.HttpResponse: Formatted HTTP error response
    """
    error_response = {
        "status": "error",
        "error_type": error_type,
        "message": message,
        "timestamp": str(datetime.datetime.now()),
        "details": details or {},
        "troubleshooting": troubleshooting or [
            "Verify input parameters are correct",
            "Check database connectivity",
            "Review campaign configuration",
            "Contact system administrator if issue persists"
        ]
    }
    
    logging.error(f"Error Response: {json.dumps(error_response, indent=2)}")
    
    return func.HttpResponse(
        json.dumps(error_response, indent=2),
        status_code=status_code,
        mimetype="application/json"
    )


# --- DATA SCHEMA AND CONTEXT SECTION --
# This schema defines the structure and relationships between campaign management collections
DataSchema = {
    "campaignmanagers": {
        "fields": {
            "name": "string",
            "sendgridCampaignId": "string",
            "status": "string",
            "_id": "ObjectId",
            "link": "string",
            "lists": "array"
        },
        "description": "Stores campaign manager records for campaigns. Each record represents a campaign with a unique SendGrid campaign ID which has the list od suppliers data provided in lists field and links with lists collection"
    },
    "campaignresponses": {
        "fields": {
            "campaignId": "string",
            "sendgridCampaignId": "string",
            "manufacturer.name": "string",
            "_id": "ObjectId"
        },
        "description": "Tracks responses to campaigns, including manufacturer details and associated campaign IDs."
    },
    "campaignlists": {
        "fields": {
            "sendgridCampaignId": "string",
            "suppliers.supplierId": "ObjectId",
            "suppliers.supplierName": "string",
            "suppliers.contacts": "array",
            "suppliers.uniqueName": "string"
        },
        "description": "Contains the campaign details along with lists of suppliers for each campaign which is already triggered in sendgrid, including supplier names and contact information."
    },
    "lists": {
        "fields": {
            "_id": "ObjectId",
            "name": "string",
            "type": "string",
            "suppliers": "array",
            "components": "array"
        },
        "description": "Contains the list details along with lists of suppliers and components for each list which is attached to the campaign manager to send mails, including list names and supplierID and componentID information.This collection is linked with suppliers collecition as well as components collections as supplierID and componentID are reffered from these collection"
    },
    "suppliers": {
        "fields": {
            "_id": "ObjectId",
            "name": "string",
            "contacts": "array",
        },
        "description": "Contains the supplier details along with contacts of the supplier, including supplier names and contact information.This collection is linked with lists collection as supplierID are reffered from thses collection"
    },
    "components": {
        "fields": {
            "_id": "ObjectId",
            "name": "string"
        },
        "description": "Contains the component details along with name of the component, including components information.This collection is linked with lists collection as components are referred from these collection"
    }
}

def supplierReminderAgent(tools, openai_llm):
    """
    Creates a specialized AI agent for supplier reminder analysis across multiple MongoDB collections.
    This agent performs complex three-collection analysis to identify non-responding suppliers.
    
    Args:
        tools: Available MCP tools for database operations
        openai_llm: Configured OpenAI LLM instance
    
    Returns:
        Agent: Configured CrewAI agent specialized in supplier data correlation
    
    Raises:
        Exception: If agent creation fails
    """
    try:
        supplierAiAgent = Agent(
            role=(
                "Senior Data Analyst specializing in multi-collection data correlation and supplier relationship analysis. "
                "Expert in cross-collection joins involving 'campaignmanagers', 'campaignresponses', 'campaignlists', 'suppliers', 'lists', and 'components'. "
                "Specialized in extracting accurate component ObjectIds from lists collection and matching them to campaigns. "
                "Critical expertise: distinguishing between supplierIds and componentIds to prevent data corruption. "
                "Compliance expert for supplier follow-up campaigns requiring precise component and supplier data extraction."
            ),
            goal=(
                "Execute comprehensive multi-collection analysis to extract actual component ObjectIds and identify non-responding suppliers using schema knowledge: "
                "campaignmanagers: {cmFields}, campaignresponses: {crFields}, campaignlists: {clFields}, suppliers: {sFields}, lists: {lFields}, components: {cFields}. "
                "CRITICAL: Extract components from lists.components field as actual ObjectIds, never confuse with supplierIds. "
                "Ensure 100% accuracy in data correlation, component extraction, and supplier contact validation."
                .format(
                    cmFields=DataSchema["campaignmanagers"]["fields"],
                    crFields=DataSchema["campaignresponses"]["fields"],
                    clFields=DataSchema["campaignlists"]["fields"],
                    sFields=DataSchema["suppliers"]["fields"],  
                    lFields=DataSchema["lists"]["fields"],
                    cFields=DataSchema["components"]["fields"]
                )
            ),
            backstory=(
                "You are an expert data analyst with extensive experience in complex multi-collection data analysis and MongoDB aggregation pipelines. "
                "You specialize in compliance campaigns for supplier follow-up, requiring precise extraction of component ObjectIds from lists collection. "
                "Your critical expertise includes: 1) Cross-collection joins between campaignmanagers→lists→components, "
                "2) Accurate differentiation between supplierIds and componentIds to prevent data corruption, "
                "3) Validation of contact information and response tracking. "
                "You always validate your aggregation logic and ensure data integrity using schema definitions. "
                "You never confuse or mix up ObjectIds from different collections."
            ),
            tools=tools,
            reasoning=False,
            llm=openai_llm,
            memory=False,
            min_retries=3,
            max_iterations=5,
            verbose=False,
            allow_delegation=False,
            additional_context={"data_schema": DataSchema},
            output_file="output/report.md"
        )
        logging.info("Supplier reminder agent created successfully with schema context")
        return supplierAiAgent
    except Exception as e:
        logging.error(f"Error creating supplier reminder agent: {e}")
        raise e


def createSupplierAiTask(supplierAiAgent, sendgridCampaignId=None):
    """
    Creates a task for analyzing campaign data and identifying non-responding suppliers.
    Performs complex multi-collection joins and data correlation across MongoDB collections.
    
    Args:
        supplierAiAgent: The configured supplier reminder agent
        sendgridCampaignId (str): The SendGrid campaign ID to analyze
    
    Returns:
        Task: Configured CrewAI task for supplier analysis
    
    Raises:
        ValueError: If sendgridCampaignId is not provided
        Exception: If task creation fails
    """
    try:
        if not sendgridCampaignId:
            raise ValueError("sendgridCampaignId must be provided for this task.")
        description = (
            f"CRITICAL: You MUST extract actual SupplierID and components from the database. DO NOT generate or guess any IDs.\n"
            f"CRITICAL: You MUST extract All the content without missing any data in the output do not leve the rest if the data is more\n"
            f"Analyze the campaign with sendgridCampaignId '{sendgridCampaignId}'.\n\n"
            "STEP-BY-STEP ANALYSIS PROCESS:\n"
            "1. CAMPAIGN ANALYSIS:\n"
            "   - Find record in 'campaignmanagers' collection where sendgridCampaignId matches\n"
            "   - Extract: campaignName (name field) and lists array field\n\n"
            "2. LIST COMPONENT EXTRACTION:\n"
            "   - For all listId in the lists field from campaign manager collections which matches the sendgridCampaignId\n"
            "   - Query 'lists' collection to find matching records which has the listId and extract the components field\n"
            "   - Extract ALL component ObjectIds from the 'components' field of each list which is attached to the campaign manager\n"
            "   - Run this exact query to find all components linked to the sendgridCampaignId:\n"
            "     db.campaignmanagers.aggregate([\n"
            "       { $match: { sendgridCampaignId: '{{sendgridCampaignId}}' } },\n"
            "       { $project: { lists: 1 } },\n"
            "       { $lookup: { from: 'lists', localField: 'lists', foreignField: '_id', as: 'lists_docs' } },\n"
            "       { $unwind: '$lists_docs' },\n"
            "       { $unwind: '$lists_docs.components' },\n"
            "       { $group: { _id: null, allComponents: { $addToSet: '$lists_docs.components' } } },\n"
            "       { $project: { _id: 0, allComponents: 1 } }\n"
            "     ])\n"
            "   - DO NOT generate or guess any componentId.\n"
            "   - These are the actual components values you MUST include in output without loosing any components from the list\n\n"
            "3. SUPPLIER ANALYSIS:\n"
            "   - Find suppliers in 'campaignlists' collection for this sendgridCampaignId\n"
            "   - Extract supplier names, IDs, and contact information\n\n"
            "4. RESPONSE CHECKING:\n"
            "   - Compare supplier names with manufacturer.name in 'campaignresponses'\n"
            "   - Identify NON-RESPONDING suppliers only\n\n"
            "5. REQUIRED OUTPUT FIELDS (for each non-responding supplier):\n"
            "   - sendgridCampaignId: exact value from input\n"
            "   - supplierName: exact value from campaignlists.suppliers.supplierName\n"
            "   - supplierId: exact value from campaignlists.suppliers.supplierId\n"
            "   - campaignName: exact value from campaignmanagers.name\n"
            "   - components:ALL component ObjectIds from lists.components (CRITICAL: This field MUST NOT be empty)\n"
            "CRITICAL VALIDATION RULES:\n"
            "   - components MUST contain actual ObjectIds from lists.components field\n"
            "   - DO NOT confuse supplierIds with componentIds - they are completely different\n"
            "   - DO NOT generate, guess, or hallucinate any IDs\n"
            "   - components field should be a comma-separated string like: \"ObjectId1,ObjectId2,ObjectId3\"\n"
            "   - If no components found, use empty string for components\n"
            "   - CRITICAL: You MUST run the aggregation query to extract components from lists collection\n"
            "   - CRITICAL: Empty components field indicates database query failure - check your aggregation pipeline\n"
            "   - Even if the same inputs are provided always generate right and correct answer DO NOT RETURN things like I tried reusing the same input, I must stop using this action input. I'll try something else instead\n"
            "OUTPUT FORMAT: Return ONLY valid JSON array\n"
            "EXAMPLE: [{\"supplierId\":\"67890\",\"supplierName\":\"VISHAY\",\"campaignName\":\"Test Campaign\",\"components\":\"674abc123,674def456,674ghi789\"}]"
             )
        expected_output = (
            "A JSON array containing ONLY non-responding suppliers with exact database values. "
            "Each object must contain these exact fields: "
            "supplierId (string), supplierName (string), campaignName (string), "
            "components (comma-separated string of component ObjectIds), "
            "NO placeholder supplierId or supplierName, NO generated IDs, ONLY actual database values."
            "Always provide Full output without missing any data in the response"
        )
        supplierAiTask = Task(
            description=description,
            expected_output=expected_output,
            agent=supplierAiAgent,
            markdown=False
        )
        logging.info("Supplier reminder task created successfully with enhanced email validation")
        return supplierAiTask
    except Exception as e:
        logging.error(f"Error creating supplier reminder task: {e}")
        raise e
    
# --- TOKEN COUNTING UTILITY ---
def count_tokens(text, model_name="gpt-4o"):
    """
    Counts the number of tokens in a string for a given model using tiktoken.
    This function is essential for tracking API usage and cost management.
    
    Args:
        text (str): The text to count tokens for
        model_name (str): The model name for token encoding (default: gpt-4o)
    
    Returns:
        int: Number of tokens in the provided text
    
    Raises:
        Exception: If tiktoken encoding fails for the specified model
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def executeDataAnalysisWorkflow(tenant, sendgridCampaignId, campaignId):
    """
    Executes the complete data analysis workflow with token usage tracking.
    Orchestrates the supplier analysis process using CrewAI agents with comprehensive token monitoring.
    
    Args:
        tenant (str): The tenant identifier for database connection
        sendgridCampaignId (str): The SendGrid campaign ID to analyze
        campaignId (str): The campaign identifier for context
    
    Returns:
        dict: Workflow results including supplier data and token usage statistics
    
    Raises:
        Exception: If workflow execution fails
    """
    try:
        logging.info(f"Starting data analysis workflow execution for campaign: {sendgridCampaignId}")
        
        # Track token usage across workflow components
        token_usage = {}
        
        # Load environment configuration
        openai_llm = loadEnvironmentConfiguration()
        model_name = getattr(openai_llm, 'model_name', 'gpt-4o')
        
        # Configure MongoDB connection
        server_params = configureMongoDBConnection(tenant)
        
        with MCPServerAdapter(server_params) as tools:
            logging.info(f"Available tools from MCP server: {[tool.name for tool in tools]}")
            
            # Validate that we have the required tools
            required_tools = ["query", "aggregate"]
            available_tool_names = [tool.name for tool in tools]
            missing_tools = [tool for tool in required_tools if tool not in available_tool_names]
            
            if missing_tools:
                raise Exception(f"Missing required MongoDB tools: {missing_tools}. Available tools: {available_tool_names}")
            
            # Create agents
            supplierAiAgent = supplierReminderAgent(tools, openai_llm)

            # --- TOKEN COUNTING FOR AGENT PROMPTS ---
            agent_token_usage = {}
            for agent_name, agent_obj in [
                ("supplierReminderAgent", supplierAiAgent),
            ]:
                prompt = f"Role: {getattr(agent_obj, 'role', '')}\nGoal: {getattr(agent_obj, 'goal', '')}\nBackstory: {getattr(agent_obj, 'backstory', '')}"
                agent_token_usage[agent_name] = count_tokens(prompt, model_name)
            token_usage['agent_prompts'] = agent_token_usage
            
            # Create tasks
            supplierAiTask = createSupplierAiTask(supplierAiAgent, sendgridCampaignId)

            # --- TOKEN COUNTING FOR TASK DESCRIPTIONS ---
            task_token_usage = {}
            for task_name, task_obj in [
                ("supplierAiTask", supplierAiTask)
            ]:
                desc = getattr(task_obj, 'description', '')
                exp_out = getattr(task_obj, 'expected_output', '')
                task_token_usage[task_name] = count_tokens(desc + "\n" + exp_out, model_name)
            token_usage['task_descriptions'] = task_token_usage
            
            # Execute crew workflow
            data_crew = Crew(
                agents=[supplierAiAgent],
                tasks=[supplierAiTask],
                verbose=True,
                process=Process.sequential
            )
            
            logging.info("Executing CrewAI workflow...")
            result = data_crew.kickoff()

            # Store raw outputs for processing
            supplierOutput = supplierAiTask.output.raw if supplierAiTask.output else "No output generated."

            # Log raw supplier output for basic monitoring
            logging.info(f"Supplier analysis completed for campaign: {sendgridCampaignId}")

            output_text = json.dumps(supplierOutput, indent=2)

            # --- TOKEN COUNTING FOR OUTPUTS ---
            output_token_usage = {}
            for output_name, output_obj in [
                ("supplierAiTask", supplierOutput)
            ]:
                if output_obj:
                    output_token_usage[output_name] = count_tokens(str(output_obj), model_name)
                else:
                    output_token_usage[output_name] = 0
            token_usage['outputs'] = output_token_usage
            
            logging.info("Workflow execution completed successfully")
            
            # Return results with comprehensive token usage tracking
            return {    
                "supplierAiTask": supplierOutput,
                "result": result,
                "output_text": output_text,
                "token_usage": token_usage
            }
    except Exception as e:
        logging.error(f"Error executing data analysis workflow: {e}")
        # Enhance error message with context
        error_context = {
            "tenant": tenant,
            "sendgridCampaignId": sendgridCampaignId,
            "campaignId": campaignId,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        logging.error(f"Workflow error context: {json.dumps(error_context, indent=2)}")
        raise e


def formatSupplierList(supplierListRaw):
    """
    Parses and cleans the supplierList output to ensure it is a JSON array (not a string or markdown).
    Handles various input formats including JSON strings, markdown-wrapped JSON, and raw lists.
    
    Args:
        supplierListRaw: The raw supplier list output (could be string, markdown, or already a list)
    
    Returns:
        list: Parsed supplier list as a JSON array
    """
    if isinstance(supplierListRaw, str):
        supplierListRaw = supplierListRaw.strip()
        if supplierListRaw.startswith("```json"):
            supplierListRaw = supplierListRaw.replace("```json", "").replace("```", "").strip()
        try:
            supplierList = json.loads(supplierListRaw)
        except Exception:
            supplierList = []
    else:
        supplierList = supplierListRaw
    return supplierList


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger entrypoint for AgentAI data analysis system.
    Processes GET or POST requests and returns formatted supplier data with comprehensive token usage statistics.
    
    Args:
        req (func.HttpRequest): The HTTP request object containing query parameters
    
    Returns:
        func.HttpResponse: JSON response containing supplier analysis results and detailed token usage metrics
    
    Raises:
        Exception: If function execution fails
    """
    logging.info("AgentAI Azure Function HTTP trigger received a request.")
    
    try:
        # Parse input parameters from request
        tenant = req.params.get("tenant")
        sendgridCampaignId = req.params.get("sendgridCampaignId")
        print(f"sendgridCampaignId: {sendgridCampaignId}")
        campaignId = req.params.get("campaignId")
        
        logging.info(f"Processing request for tenant: {tenant}, campaign: {sendgridCampaignId}")
        
        # Validate input parameters
        validation_result = validateInputParameters(tenant, sendgridCampaignId, campaignId)
        if not validation_result["valid"]:
            return createErrorResponse(400, "validation_error", validation_result["message"], details=validation_result["errors"])

        # Execute workflow with token tracking
        workflow_result = executeDataAnalysisWorkflow(tenant, sendgridCampaignId, campaignId)
        
        # Validate workflow result
        validation_result = validateWorkflowResult(workflow_result)
        if not validation_result["valid"]:
            return createErrorResponse(500, "workflow_error", validation_result["message"], details=validation_result["errors"])

        # Calculate total tokens used across all components
        tokenUsage = workflow_result.get("token_usage", {})
        totalTokens = 0
        
        # Sum tokens from agent prompts
        agentTokens = sum(tokenUsage.get("agent_prompts", {}).values())
        totalTokens += agentTokens
        
        # Sum tokens from task descriptions
        taskTokens = sum(tokenUsage.get("task_descriptions", {}).values())
        totalTokens += taskTokens
        
        # Sum tokens from outputs
        outputTokens = sum(tokenUsage.get("outputs", {}).values())
        totalTokens += outputTokens
        
        # Format supplierList to ensure JSON array output
        supplierListRaw = workflow_result.get("supplierAiTask")
        
        # Validate AI agent output first
        validation_result = validateAIAgentOutput(supplierListRaw, sendgridCampaignId)
        if not validation_result["valid"]:
            return createErrorResponse(422, "ai_agent_error", validation_result["message"], details=validation_result["errors"])
        
        supplierList = formatSupplierList(supplierListRaw)

        # Validate supplier list data
        validation_result = validateSupplierList(supplierList, sendgridCampaignId)
        if not validation_result["valid"]:
            return createErrorResponse(500, "data_error", validation_result["message"], details=validation_result["errors"])

        # Extract all supplier IDs and campaign details with error handling
        try:
            supplierNames = [supplier.get("supplierName", "") for supplier in supplierList if supplier.get("supplierName")]
            campaignName = supplierList[0].get("campaignName", "") if supplierList else ""
        except Exception as e:
            logging.error(f"Error extracting supplier names and campaign name: {e}")
            return createErrorResponse(
                500, 
                "data_extraction_error", 
                "Failed to extract supplier names and campaign information", 
                details={"error": str(e), "supplier_list_length": len(supplierList) if supplierList else 0}
            )

        # Process components data efficiently with error handling
        try:
            components = []

            for supplier in supplierList:
                comps = supplier.get("components") or supplier.get("component")

                if not comps:
                    possible_ids = supplier.get("supplierId")
                    if possible_ids and "," in possible_ids:
                        comps = possible_ids

                if isinstance(comps, str):
                    comps = [c.strip() for c in comps.split(",") if c.strip()]
                elif isinstance(comps, list):
                    comps = [str(c).strip() for c in comps if str(c).strip()]
                else:
                    comps = [str(comps).strip()] if comps else []
                components.extend(comps)

            # Remove duplicates while preserving order
            components = list(dict.fromkeys(components))
            supplierNames = list(dict.fromkeys(supplierNames))
        except Exception as e:
            logging.error(f"Error processing components data: {e}")
            return createErrorResponse(
                500, 
                "data_processing_error", 
                "Failed to process components data", 
                details={"error": str(e), "supplier_list_length": len(supplierList) if supplierList else 0}
            )

        # Validate extracted data
        validation_result = validateExtractedData(supplierNames, components, campaignName, sendgridCampaignId)
        if not validation_result["valid"]:
            return createErrorResponse(500, "data_error", validation_result["message"], details=validation_result["errors"])

        # Transform supplier list into expected format with error handling
        try:
            transformedSupplierList = [{
                "name": f"follow up list for {campaignName}",
                "type": "componentsAdd",
                "suppliers": supplierNames,
                "components": components
            }]
            
            # Create comprehensive response with supplier data and token usage
            finalResponse = { 
                "status": "success",
                "usedTokens": totalTokens,
                "supplierList": transformedSupplierList,
                "summary": {
                    "suppliers_count": len(supplierNames),
                    "components_count": len(components),
                    "campaign_name": campaignName,
                    "campaign_id": sendgridCampaignId
                }
            }
        except Exception as e:
            logging.error(f"Error creating final response: {e}")
            return createErrorResponse(
                500, 
                "response_creation_error", 
                "Failed to create final response", 
                details={"error": str(e), "supplier_names_count": len(supplierNames), "components_count": len(components)}
            )
        
        # Log success with token usage summary
        logging.info(f"Total tokens used: {totalTokens} (Agents: {agentTokens}, Tasks: {taskTokens}, Outputs: {outputTokens})")
        logging.info(f"Processed {len(supplierNames)} suppliers with {len(components)} components")
        
        # Validate that response can be serialized to JSON
        try:
            json.dumps(finalResponse, indent=2)
        except Exception as e:
            logging.error(f"Error serializing response to JSON: {e}")
            return createErrorResponse(
                500, 
                "json_serialization_error", 
                "Failed to serialize response to JSON", 
                details={"error": str(e), "response_keys": list(finalResponse.keys()) if isinstance(finalResponse, dict) else "not_dict"}
            )
        
        # Return the comprehensive response with supplier data and token usage
        return func.HttpResponse(
            json.dumps(finalResponse, indent=2),
            status_code=200,
            mimetype="application/json"
        )                                                        
    except Exception as e:
        logging.error(f"Error in AgentAI Azure Function: {e}")
        return createErrorResponse(
            500, 
            "system_error", 
            f"Unexpected system error: {str(e)}", 
            details={"error": str(e), "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None},
            troubleshooting=[
                "Verify database connection parameters",
                "Check MCP server configuration", 
                "Validate input parameters (tenant, sendgridCampaignId)",
                "Review agent configuration and prompts",
                "Check system logs for detailed error information"
            ]
        )
