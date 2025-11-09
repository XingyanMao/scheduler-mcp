# test_ai.py
import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from mcp_scheduler.config import Config
from mcp_scheduler.executor import Executor
from mcp_scheduler.task import Task, TaskType

async def test_ai():
    """测试AI功能"""
    print("Testing AI functionality...")
    
    # 使用相同的配置
    config = Config()
    executor = Executor(config)
    
    # 创建一个测试任务
    test_task = Task(
        name="Test AI Task",
        schedule="* * * * *",  # 每分钟执行一次
        type=TaskType.AI,
        prompt="请用中文回答：什么是人工智能？"
    )
    
    print(f"Testing with prompt: {test_task.prompt}")
    
    # 使用公共方法执行任务
    execution = await executor.execute_task(test_task)
    
    if execution.error:
        print(f"❌ AI task failed: {execution.error}")
    else:
        print(f"✅ AI task successful: {execution.output}")

if __name__ == "__main__":
    asyncio.run(test_ai())


    