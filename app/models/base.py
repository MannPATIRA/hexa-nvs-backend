from sqlalchemy import Column, String, Integer, Numeric, Boolean, DateTime, ForeignKey, Text, ARRAY, text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    currency = Column(String, default="BRL")
    language = Column(String, default="pt-BR")
    po_number_prefix = Column(String, default="PO")
    po_next_number = Column(Integer, default=1)
    stale_price_days = Column(Integer, default=7)
    erp_type = Column(String, default="none")
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    email = Column(String, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, default="buyer")
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    last_login_at = Column(DateTime(timezone=True))

class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    name = Column(String, nullable=False)
    contact_emails = Column(ARRAY(String), default=[])
    default_terms = Column(String)
    categories = Column(ARRAY(String), default=[])
    reliability_score = Column(Numeric(4, 3), default=0.5)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class Product(Base):
    __tablename__ = "products"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    canonical_name = Column(String, nullable=False)
    category = Column(String)
    unit = Column(String, default="caixa")
    pack_size = Column(Integer)
    embedding = Column(Vector(512))  # text-embedding-3-small @ 512 dims
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class ProductAlias(Base):
    __tablename__ = "product_aliases"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey("suppliers.id"))
    raw_description = Column(String, nullable=False)
    normalised_description = Column(String, nullable=False)
    supplier_code = Column(String)
    source = Column(String, default="manual")
    confirmed = Column(Boolean, default=False)
    embedding = Column(Vector(512))  # text-embedding-3-small @ 512 dims
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class PriceRecord(Base):
    __tablename__ = "price_records"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey("suppliers.id"), nullable=False)
    unit_price = Column(Numeric(12, 2), nullable=False)
    currency = Column(String, default="BRL")
    payment_terms = Column(String)
    source_type = Column(String, nullable=False)
    source_document_id = Column(UUID(as_uuid=True))
    observed_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class Document(Base):
    __tablename__ = "documents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    email_message_id = Column(String)
    sender_email = Column(String)
    subject = Column(String)
    document_type = Column(String, nullable=False)
    processing_status = Column(String, default="pending")
    supplier_id = Column(UUID(as_uuid=True), ForeignKey("suppliers.id"))
    items_extracted = Column(Integer, default=0)
    items_matched = Column(Integer, default=0)
    items_unresolved = Column(Integer, default=0)
    raw_file_path = Column(String)
    error_detail = Column(Text)
    processed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class PendingMatch(Base):
    __tablename__ = "pending_matches"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey("suppliers.id"), nullable=False)
    raw_description = Column(String, nullable=False)
    supplier_code = Column(String)
    extracted_price = Column(Numeric(12, 2))
    extracted_terms = Column(String)
    suggested_product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"))
    confidence = Column(Numeric(4, 3), default=0.0)
    match_strategy = Column(String, default="none")
    status = Column(String, default="pending")
    resolved_product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    resolved_at = Column(DateTime(timezone=True))

class ProcurementQueueItem(Base):
    __tablename__ = "procurement_queue"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    quantity_needed = Column(Integer, nullable=False)
    source_type = Column(String, nullable=False)
    source_detail = Column(String)
    urgency = Column(String, default="medium")
    status = Column(String, default="pending")
    assigned_supplier_id = Column(UUID(as_uuid=True), ForeignKey("suppliers.id"))
    agreed_price = Column(Numeric(12, 2))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    po_number = Column(String, nullable=False)
    supplier_id = Column(UUID(as_uuid=True), ForeignKey("suppliers.id"), nullable=False)
    status = Column(String, default="draft")
    total_value = Column(Numeric(14, 2), default=0.0)
    payment_terms = Column(String)
    pdf_file_path = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class PoLineItem(Base):
    __tablename__ = "po_line_items"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    po_id = Column(UUID(as_uuid=True), ForeignKey("purchase_orders.id"), nullable=False)
    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Numeric(12, 2), nullable=False)
    line_total = Column(Numeric(14, 2), nullable=False)
    procurement_queue_id = Column(UUID(as_uuid=True), ForeignKey("procurement_queue.id"))
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(UUID(as_uuid=True))
    details = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=text("NOW()"))