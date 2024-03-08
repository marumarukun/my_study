from typing import Optional

from pydantic import BaseModel, Field


class TaskBase(BaseModel):
    # id: int
    title: Optional[str] = Field(None, example="クリーニングを取りに行く", description="タスクのタイトル")
    # done: bool = Field(False, description="完了フラグ")


class TaskCreate(TaskBase):
    pass


class TaskCreateResponse(TaskBase):
    id: int
    
    class Config:
        orm_mode = True


class Task(TaskBase):
    id: int
    done: bool = Field(False, description="完了フラグ")
    
    class Config:
        orm_mode = True
    


