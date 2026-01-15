import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras
from langchain_core.messages import ToolMessage
import json

load_dotenv()

SERVERS = { 
    "Arithmetic MCP Server": {
      "transport":"stdio",
      "command": "C:\\Users\\panka\\anaconda3\\Scripts\\uv.exe",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "C:\\Users\\panka\\OneDrive\\Desktop\\local mcp maths server\\main.py"
      ]
    },

    # "Expense Tracker remote server": {
    #     "transport": "streamable_http",  # if this fails, try "sse"                 # hobby plan not allowed remote server
    #     "url": "https://uninterested-peach-panda.fastmcp.app/mcp"
    # }
    
}

async def main():
    
    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    named_tools = {}
    for tool in tools:
        named_tools[tool.name] = tool

    # print("named_tools:", named_tools)
    print("Available tools:", named_tools.keys())

    llm = ChatCerebras(model="llama-3.3-70b")

    llm_with_tools=llm.bind_tools(tools)

    prompt='what is the the result 8 + 78 ?'
    response=await llm_with_tools.ainvoke(prompt)

    if not getattr(response, 'tool_calls', None): 
        print("\nLLM reply: " , response.content)
        return

    # print('Response: ', response)

    tool_message=[]
    for tc in response.tool_calls: 
        selected_tool=tc['name']
        selected_tool_args=tc['args']
        selected_tool_id=tc['id']

        tool_response=await named_tools[selected_tool].ainvoke(selected_tool_args)
        tool_result=tool_response[0]['text']
        # tool_message=ToolMessage(content=tool_result, tool_name=selected_tool, tool_call_id=selected_tool_id)
        tool_message=ToolMessage(content=tool_result, tool_call_id=selected_tool_id)
                # we can skip tool name but tool id is mandatory to pass here


    final_response=await llm_with_tools.ainvoke([prompt, response, tool_message])

    print(f"final response: {final_response.content}")

if __name__ == '__main__':
    asyncio.run(main())