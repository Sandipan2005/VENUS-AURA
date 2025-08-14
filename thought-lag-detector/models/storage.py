

import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON, func
from datetime import datetime

db = SQLAlchemy()

class Session(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	client_id = db.Column(db.String(64), index=True)
	started_at = db.Column(db.DateTime, default=datetime.utcnow)
	is_baseline = db.Column(db.Boolean, default=False)

class PromptResult(db.Model):
	def __init__(self, session_id=None, prompt_idx=None, prompt_text=None, audio_path=None, features=None, stress_score=None, focus_score=None, created_at=None):
		self.session_id = session_id
		self.prompt_idx = prompt_idx
		self.prompt_text = prompt_text
		self.audio_path = audio_path
		self.features = features
		self.stress_score = stress_score
		self.focus_score = focus_score
		self.created_at = created_at if created_at is not None else datetime.utcnow()
	id = db.Column(db.Integer, primary_key=True)
	session_id = db.Column(db.Integer, db.ForeignKey("session.id"))
	prompt_idx = db.Column(db.Integer)
	prompt_text = db.Column(db.Text)
	audio_path = db.Column(db.Text)
	features = db.Column(JSON)
	stress_score = db.Column(db.Float)
	focus_score = db.Column(db.Float)
	created_at = db.Column(db.DateTime, default=datetime.utcnow)

class BaselineStat(db.Model):
	__tablename__ = 'baseline_stat'
	id = db.Column(db.Integer, primary_key=True)
	client_id = db.Column(db.String(64), index=True, unique=True)
	feature_means = db.Column(JSON)
	feature_stds = db.Column(JSON)
	count = db.Column(db.Integer, default=0)
	updated_at = db.Column(db.DateTime, default=datetime.utcnow)

def ensure_dirs(app):
	os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
