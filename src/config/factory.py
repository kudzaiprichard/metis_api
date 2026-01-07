"""
Flask application factory.
Creates and configures the Flask application with all necessary extensions and settings.
"""

import os
import logging as python_logging
from datetime import timedelta
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt

from src.modules.authentication.internal.ml_engineer_initializer import MLEngineerInitializer
from src.shared.data.database import db
from src.shared.exceptions.error_handlers import register_error_handlers
from src.shared.data.neo4j.neo4j_manager import get_neo4j_manager
from src.shared.ml.service_initializer import get_ml_service


def create_app(flask_app, configs, blueprints=None):
    """
    Create and configure the Flask application.

    Args:
        flask_app: Flask application instance
        configs: Dictionary of configuration sections
        blueprints: List of blueprints to register (optional)

    Returns:
        Configured Flask application
    """

    if blueprints is None:
        blueprints = []

    # Initialize Flask app
    app = flask_app

    # ============================================
    # CORS Configuration
    # ============================================
    cors_origins = configs.get("server").cors_allowed_origins

    # Convert to list if it's a comma-separated string
    if isinstance(cors_origins, str):
        cors_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

    CORS(app, resources={
        r"/api/*": {
            "origins": cors_origins,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })

    # Validate required database config
    if not configs.get("database").uri:
        raise ValueError("Database URI is not set in the configuration.")

    # ============================================
    # Application Metadata
    # ============================================
    app.config['APP_NAME'] = configs.get("application").name
    app.config['APP_DESCRIPTION'] = configs.get("application").description
    app.config['APP_VERSION'] = configs.get("application").version

    # ============================================
    # Server Configuration
    # ============================================
    app.config['DEBUG'] = configs.get("server").debug

    # ============================================
    # Database Configuration
    # ============================================
    app.config['SQLALCHEMY_DATABASE_URI'] = configs.get("database").uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = configs.get("database").track_modifications
    app.config['SQLALCHEMY_POOL_SIZE'] = configs.get("database").pool_size
    app.config['SQLALCHEMY_MAX_OVERFLOW'] = configs.get("database").max_overflow
    app.config['SQLALCHEMY_POOL_TIMEOUT'] = configs.get("database").pool_timeout
    app.config['SQLALCHEMY_POOL_RECYCLE'] = configs.get("database").pool_recycle
    app.config['SQLALCHEMY_ECHO'] = configs.get("database").echo
    app.config['SQLALCHEMY_POOL_PRE_PING'] = configs.get("database").pool_pre_ping

    # ============================================
    # Initialize SQLAlchemy
    # ============================================
    db.init_app(app)

    # Create all tables
    with app.app_context():
        db.create_all()
        MLEngineerInitializer.initialize()
        app.logger.info("Database tables created successfully")

    # ============================================
    # JWT Configuration
    # ============================================
    jwt_config = configs.get("security").jwt

    app.config['JWT_SECRET_KEY'] = jwt_config.secret_key
    app.config['JWT_REFRESH_SECRET_KEY'] = jwt_config.refresh_secret_key
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(seconds=jwt_config.access_token_expires)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(seconds=jwt_config.refresh_token_expires)
    app.config['JWT_ALGORITHM'] = jwt_config.algorithm
    app.config['JWT_TOKEN_LOCATION'] = jwt_config.token_location

    # Initialize JWT Manager
    jwt = JWTManager(app)

    # ============================================
    # Password Hashing Configuration
    # ============================================
    password_config = configs.get("security").password
    app.config['BCRYPT_LOG_ROUNDS'] = password_config.bcrypt_rounds

    # Initialize Bcrypt
    bcrypt = Bcrypt(app)

    # ============================================
    # Logging Configuration
    # ============================================
    logging_config = configs.get("logging")

    # Configure Python logging
    log_level = getattr(python_logging, logging_config.level.upper(), python_logging.INFO)

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(logging_config.file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    python_logging.basicConfig(
        level=log_level,
        format=logging_config.format,
        handlers=[
            python_logging.FileHandler(logging_config.file_path),
            python_logging.StreamHandler()
        ]
    )

    # Set Flask's logger to use the same configuration
    app.logger.setLevel(log_level)

    # Silence verbose third-party loggers
    python_logging.getLogger('neo4j').setLevel(python_logging.ERROR)
    python_logging.getLogger('neo4j.pool').setLevel(python_logging.ERROR)
    python_logging.getLogger('neo4j.io').setLevel(python_logging.ERROR)
    python_logging.getLogger('neo4j.routing').setLevel(python_logging.ERROR)

    # ============================================
    # Neo4j Configuration
    # ============================================
    neo4j_config = configs.get("integrations").graph_database

    app.config['NEO4J_CONFIG'] = {
        'uri': neo4j_config.uri,
        'username': neo4j_config.username,
        'password': neo4j_config.password
    }

    # Initialize Neo4j Manager
    with app.app_context():
        neo4j_manager = get_neo4j_manager()
        neo4j_manager.initialize(app)

    # ============================================
    # ML Configuration
    # ============================================
    ml_config = configs.get("integrations").ml_model
    gemini_config = configs.get("integrations").ai.gemini

    app.config['ML_CONFIG'] = {
        'models_dir': ml_config.models_dir,
        'scaler_path': ml_config.scaler_path,
        'preprocessing_metadata_path': ml_config.preprocessing_metadata_path,
        'model_version': ml_config.model_version if ml_config.model_version else None,
        'auto_activate': ml_config.auto_activate,
        'verbose': ml_config.verbose
    }

    app.config['GEMINI_CONFIG'] = {
        'api_key': gemini_config.api_key,
        'model': gemini_config.model
    }

    # Initialize ML Services
    with app.app_context():
        ml_service = get_ml_service()
        ml_service.initialize(app)

    # ============================================
    # Register Blueprints
    # ============================================
    for blueprint in blueprints:
        app.register_blueprint(blueprint)
        app.logger.info(f"Registered blueprint: {blueprint.name}")

    # ============================================
    # Register Global Error Handlers
    # ============================================
    register_error_handlers(app)
    app.logger.info("Registered global error handlers")

    # ============================================
    # Application Ready
    # ============================================
    app.logger.info(f"{app.config['APP_NAME']} v{app.config['APP_VERSION']} initialized successfully")

    return app