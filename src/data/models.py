from __future__ import annotations

from sqlalchemy import Column, Integer, Float, DateTime, String, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from src.data.db import Base


class Batch(Base):
    __tablename__ = "batches"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    age_days = Column(Float)
    chick_weight_g = Column(Float)
    temp_c = Column(Float)
    humidity_pct = Column(Float)
    stocking_density = Column(Float)
    vaccine_score = Column(Float)
    breed_index = Column(Float)
    housing_quality = Column(Float)
    feed_protein_pct = Column(Float)
    water_quality = Column(Float)
    management_index = Column(Float)
    flock_size = Column(Float)
    feed_price_usd_kg = Column(Float)
    sale_price_usd_kg = Column(Float)
    energy_cost_usd = Column(Float)

    predictions = relationship("Prediction", back_populates="batch", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(Integer, ForeignKey("batches.id"))
    predicted_at = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String(255), nullable=True)

    final_weight_kg = Column(Float)
    mortality_rate_pct = Column(Float)
    avg_daily_gain_g = Column(Float)
    feed_intake_kg = Column(Float)
    fcr = Column(Float)
    annual_revenue_usd = Column(Float)

    batch = relationship("Batch", back_populates="predictions")
