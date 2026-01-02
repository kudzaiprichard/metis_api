import sys
import os
from flask import Flask
from dotenv import load_dotenv
from src.config.factory import create_app
from src.modules.authentication.presentation.controllers.auth_controller import auth_bp
from src.modules.authentication.presentation.controllers.user_controller import user_bp
from src.modules.patients.presentation.controller.patient_controller import patients_bp
from src.modules.recommendation.presentation.controllers.recommendation_controller import recommendation_bp


def load_environment():
    """Load the appropriate .env file based on FLASK_ENV."""
    env = os.getenv('FLASK_ENV', 'development')
    env_file = f'.env.{env}'

    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"✅ Loaded environment: {env} from {env_file}")
    else:
        print(f"✅ Running in {env} mode - using system environment variables")

    return env


def ensure_config_ready():
    """Ensure config.py runs and config.pyi is generated before imports."""
    try:
        from src.config.config import config_sections
        print("✅ Config system initialized")
        return bool(config_sections)
    except Exception as e:
        print(f"❌ Error initializing config: {e}")
        return False


def create_application():
    """Create and return Flask application for Gunicorn."""

    print("=== Initializing Configuration System ===")

    current_env = load_environment()
    print(f"📦 Environment: {current_env}")
    print(f"📦 App Version: {os.getenv('APP_VERSION', 'unknown')}")

    if not ensure_config_ready():
        print("Failed to initialize config system. Exiting.")
        sys.exit(1)

    try:
        from src.config.config import (
            application,
            security,
            server,
            database,
            integrations,
            logging
        )

        configs = {
            "application": application,
            "security": security,
            "server": server,
            "database": database,
            "integrations": integrations,
            "logging": logging,
        }

        print("✅ Loaded configs:", list(configs.keys()))

    except ImportError as e:
        print(f"❌ Failed to import configs: {e}")
        sys.exit(1)

    print("\n=== Starting Flask Application ===")

    blueprints = [auth_bp, user_bp, patients_bp, recommendation_bp]

    try:
        flask_app = create_app(
            Flask(__name__),
            configs=configs,
            blueprints=blueprints
        )

        print("✅ Flask application created successfully")
        print("✅ Blueprints registered:", [bp.name for bp in blueprints])

        return flask_app

    except AttributeError as e:
        print(f"❌ Configuration error: {e}")
        print("Make sure your config.yaml file has all required sections.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)


# Create the app instance (Gunicorn will use this)
app = create_application()

# For direct execution with Flask development server
if __name__ == '__main__':
    from src.config.config import server

    port = int(os.getenv('PORT', server.port))

    print(f"🚀 Starting development server with Flask")
    print(f"   Host: {server.ip}")
    print(f"   Port: {port}")
    print(f"   Debug: {server.debug}")

    app.run(host=server.ip, port=port, debug=server.debug)