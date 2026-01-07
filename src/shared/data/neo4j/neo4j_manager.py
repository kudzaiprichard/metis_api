"""
Neo4j Connection Manager - Singleton for shared Neo4j access across the application.

CRITICAL: Neo4j is REQUIRED for system operation. Application will not start without it.
"""

from src.shared.data.neo4j.neo4j_graph_database import Neo4jGraphDatabase
import logging

logger = logging.getLogger(__name__)


class Neo4jManager:
    """
    Singleton manager for Neo4j database connections.

    Provides centralized access to Neo4j for:
    - ML explainability (clinical knowledge, guidelines)
    - Patient data queries
    - Treatment outcome analysis
    - Similar patient case finding
    - Clinical analytics

    CRITICAL: This is a REQUIRED service. Application startup will fail if Neo4j is unavailable.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._neo4j_db = None
            self._app = None
            self._config = None
            self._initialized = True

    def initialize(self, app):
        """
        Initialize Neo4j connection from Flask app config.

        CRITICAL: This method will raise RuntimeError if Neo4j connection fails.
        The application startup will be aborted.

        Args:
            app: Flask application instance

        Raises:
            RuntimeError: If Neo4j configuration missing or connection fails
        """
        self._app = app

        # Validate configuration exists
        neo4j_config = app.config.get('NEO4J_CONFIG')
        if not neo4j_config:
            raise RuntimeError("Neo4j configuration not found in app config")

        self._config = neo4j_config

        # Log connection attempt
        app.logger.info("=" * 80)
        app.logger.info("INITIALIZING NEO4J CONNECTION MANAGER")
        app.logger.info("=" * 80)
        app.logger.info(f"Neo4j URI: {neo4j_config['uri']}")
        app.logger.info(f"Neo4j Username: {neo4j_config['username']}")

        try:
            # Create Neo4j database instance
            self._neo4j_db = Neo4jGraphDatabase(
                uri=neo4j_config['uri'],
                username=neo4j_config['username'],
                password=neo4j_config['password']
            )

            # Attempt connection
            if not self._neo4j_db.connect():
                raise RuntimeError(
                    f"Failed to connect to Neo4j at {neo4j_config['uri']}. "
                    "Ensure Neo4j is running and credentials are correct."
                )

            # Verify connection with test query
            self._verify_connection()

            # Log success
            app.logger.info("Neo4j Connection Manager initialized successfully")
            app.logger.info(f"Connected to: {neo4j_config['uri']}")
            app.logger.info("=" * 80)

        except Exception as e:
            # Clean error message - no verbose troubleshooting
            self._neo4j_db = None
            raise RuntimeError(f"Neo4j initialization failed: {str(e)}")

    def _verify_connection(self):
        """
        Verify Neo4j connection with a test query.

        Raises:
            RuntimeError: If verification query fails
        """
        try:
            driver = self._neo4j_db.driver
            with driver.session() as session:
                result = session.run("RETURN 'Connection verified' AS message")
                message = result.single()['message']
                self._app.logger.info(f"Neo4j verification: {message}")
        except Exception as e:
            raise RuntimeError(f"Neo4j connection verification failed: {str(e)}")

    def get_database(self) -> Neo4jGraphDatabase:
        """
        Get Neo4j database instance.

        Returns:
            Neo4jGraphDatabase instance

        Raises:
            RuntimeError: If Neo4j not initialized (should never happen in production)
        """
        if self._neo4j_db is None:
            raise RuntimeError(
                "Neo4j database not initialized. "
                "This should not happen if application startup succeeded."
            )
        return self._neo4j_db

    def is_available(self) -> bool:
        """
        Check if Neo4j is available.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._neo4j_db is not None and self._neo4j_db.is_connected()

    def get_driver(self):
        """
        Get raw Neo4j driver for custom queries.

        Use this for complex queries not covered by Neo4jGraphDatabase methods.

        Returns:
            Neo4j driver instance

        Raises:
            RuntimeError: If Neo4j not initialized
        """
        if self._neo4j_db is None:
            raise RuntimeError("Neo4j database not initialized")
        return self._neo4j_db.driver

    def get_connection_info(self) -> dict:
        """
        Get Neo4j connection information.

        Returns:
            dict: Connection details (URI, username, status)
        """
        if self._config is None:
            return {
                'status': 'not_initialized',
                'uri': None,
                'username': None,
                'connected': False
            }

        return {
            'status': 'connected' if self.is_available() else 'disconnected',
            'uri': self._config['uri'],
            'username': self._config['username'],
            'connected': self.is_available()
        }

    def health_check(self) -> dict:
        """
        Perform comprehensive health check on Neo4j connection.

        Returns:
            dict: Health check results with status and details
        """
        if not self.is_available():
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': 'Neo4j not available'
            }

        try:
            driver = self._neo4j_db.driver

            # Test query
            with driver.session() as session:
                start_time = session.run("RETURN timestamp()").single()[0]

                # Count nodes (basic health check)
                node_count_result = session.run("MATCH (n) RETURN count(n) AS count")
                node_count = node_count_result.single()['count']

                # Count relationships
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
                rel_count = rel_count_result.single()['count']

            return {
                'status': 'healthy',
                'connected': True,
                'uri': self._config['uri'],
                'node_count': node_count,
                'relationship_count': rel_count,
                'timestamp': start_time
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e)
            }

    def close(self):
        """
        Close Neo4j connection.

        Should be called during application shutdown.
        """
        if self._neo4j_db:
            self._neo4j_db.close()
            if self._app:
                self._app.logger.info("Neo4j Manager connection closed")
            self._neo4j_db = None


def get_neo4j_manager():
    """
    Get Neo4j manager singleton.

    Returns:
        Neo4jManager instance
    """
    return Neo4jManager()