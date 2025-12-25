import os
import logging as python_logging
from datetime import timedelta
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_bcrypt import Bcrypt
from src.shared.data.database import db

def create_app(flask_app,
               configs,
               blueprints=None,
               error_handlers=None):
    """Create and configure the Flask application using passed config parameters."""

    if blueprints is None:
        blueprints = []
    if error_handlers is None:
        error_handlers = []

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

    # Create all tables (including sessions table)
    with app.app_context():
        db.create_all()
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

    # Initialize JWT
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
            python_logging.StreamHandler()  # Also log to console
        ]
    )

    # Set Flask's logger to use the same configuration
    app.logger.setLevel(log_level)

    # ============================================
    # Register Blueprints (controllers)
    # ============================================
    for blueprint in blueprints:
        app.register_blueprint(blueprint)
        app.logger.info(f"Registered blueprint: {blueprint.name}")

    # ============================================
    # Register custom error handling
    # ============================================
    for error, handler in error_handlers:
        app.register_error_handler(error, handler)

    app.logger.info(f">> {app.config['APP_NAME']} v{app.config['APP_VERSION']} initialized successfully")

    return app