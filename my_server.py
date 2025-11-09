from fastmcp import FastMCP

# 创建一个FastMCP服务器实例，并给它起个名字
mcp = FastMCP("My First Tools")

# 使用装饰器将普通Python函数声明为AI可用的工具
@mcp.tool
def add_numbers(a: int, b: int) -> int:
    """将两个数字相加并返回结果。"""
    return a + b

# 启动服务器
if __name__ == "__main__":
    mcp.run(transport="stdio")