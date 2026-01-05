"""
Database models for conversation history and knowledge management.
Based on Dify's SQLAlchemy patterns.
"""
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    Boolean,
    ForeignKey,
    JSON,
    create_engine,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    Session,
)
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class Conversation(Base):
    """Conversation session model."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(100), nullable=True, index=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    meta_info = Column(JSON, nullable=True)

    # Relationship
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, nullable=True)
    meta_info = Column(JSON, nullable=True)

    # Relationship
    conversation = relationship("Conversation", back_populates="messages")


class KnowledgeDocument(Base):
    """Knowledge base document model."""

    __tablename__ = "knowledge_documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=True)
    file_size = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_info = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)


class Feedback(Base):
    """User feedback model."""

    __tablename__ = "feedbacks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    message_id = Column(String(36), ForeignKey("messages.id"), nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5 star rating
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentSession(Base):
    """Agent session for tracking agent conversations with tool usage."""

    __tablename__ = "agent_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    state = Column(String(50), default="active")  # active, completed, error
    plan = Column(JSON, nullable=True)  # Task plan if any
    memory_summary = Column(Text, nullable=True)  # Summarized memory
    thoughts = Column(JSON, nullable=True)  # Reasoning steps
    total_iterations = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_info = Column(JSON, nullable=True)

    # Relationships
    conversation = relationship("Conversation", backref="agent_sessions")
    tool_executions = relationship(
        "ToolExecution",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ToolExecution.created_at",
    )


class ToolExecution(Base):
    """Record of tool execution during agent sessions."""

    __tablename__ = "tool_executions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(
        String(36),
        ForeignKey("agent_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tool_name = Column(String(100), nullable=False)
    input_params = Column(JSON, nullable=True)
    output = Column(JSON, nullable=True)
    status = Column(String(20), default="pending")  # pending, running, success, error
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("AgentSession", back_populates="tool_executions")


class DatabaseManager:
    """Database connection and session management."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.is_async = database_url.startswith("sqlite+aiosqlite")

        if self.is_async:
            self.engine = create_async_engine(
                database_url,
                echo=False,
                future=True,
            )
            self.async_session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        else:
            self.engine = create_engine(database_url, echo=False)
            self.session_factory = sessionmaker(bind=self.engine)

    async def init_db(self):
        """Initialize database tables."""
        if self.is_async:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """Get synchronous session."""
        if self.is_async:
            raise RuntimeError("Use get_async_session for async database")
        return self.session_factory()

    async def get_async_session(self) -> AsyncSession:
        """Get async session."""
        if not self.is_async:
            raise RuntimeError("Database is not configured for async")
        return self.async_session_factory()


# Repository classes for database operations
class ConversationRepository:
    """Repository for conversation operations."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, user_id: Optional[str] = None, title: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        conv = Conversation(user_id=user_id, title=title)
        self.session.add(conv)
        self.session.commit()
        return conv

    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self.session.query(Conversation).filter_by(id=conversation_id).first()

    def get_user_conversations(self, user_id: str, limit: int = 20) -> List[Conversation]:
        """Get user's conversations."""
        return (
            self.session.query(Conversation)
            .filter_by(user_id=user_id, is_active=True)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta_info: Optional[dict] = None,
    ) -> Message:
        """Add a message to conversation."""
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            meta_info=meta_info,
        )
        self.session.add(msg)

        # Update conversation timestamp
        conv = self.get_by_id(conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()

        self.session.commit()
        return msg

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> List[Message]:
        """Get conversation messages."""
        return (
            self.session.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .all()
        )

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        conv = self.get_by_id(conversation_id)
        if conv:
            self.session.delete(conv)
            self.session.commit()
            return True
        return False
