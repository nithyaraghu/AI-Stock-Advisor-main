import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from config import settings

DATABASE_URL = settings['database_url']

# ── Connect to PostgreSQL ──────────────────────────────────────
def get_connection():
    """Get a raw psycopg2 connection. Always close after use."""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def get_cursor(conn):
    """Get a cursor that returns rows as dicts (easier to work with)."""
    return conn.cursor(cursor_factory=RealDictCursor)

# ── Create all tables on first run ────────────────────────────
def init_db():
    """
    Run this once to create all tables.
    Replaces Cosmos DB container creation.
    Called automatically when app.py starts.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Enable pgvector extension (replaces ChromaDB)
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')

    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            email       TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            name        TEXT,
            preferences JSONB DEFAULT '{}',
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Portfolio holdings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_holdings (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
            symbol     TEXT NOT NULL,
            quantity   FLOAT NOT NULL,
            avg_cost   FLOAT NOT NULL,
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, symbol)
        );
    """)

    # Stock embeddings — replaces ChromaDB
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_embeddings (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            symbol     TEXT NOT NULL,
            data_text  TEXT NOT NULL,
            embedding  vector(1536),
            metadata   JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # News embeddings with sentiment
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news_embeddings (
            id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            symbol          TEXT NOT NULL,
            headline        TEXT NOT NULL,
            content         TEXT,
            embedding       vector(1536),
            sentiment_score FLOAT DEFAULT 0.0,
            published_at    TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Agent memory — long term memory per user
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
            memory_type TEXT DEFAULT 'fact',
            content     TEXT NOT NULL,
            embedding   vector(1536),
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Chat history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            user_id    UUID REFERENCES users(id) ON DELETE CASCADE,
            role       TEXT NOT NULL,
            message    TEXT NOT NULL,
            agent_used TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Price cache
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            symbol     TEXT PRIMARY KEY,
            price      FLOAT,
            change_pct FLOAT,
            ohlcv      JSONB,
            fetched_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)

    # Indexes for fast vector search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_stock_embeddings_vector
        ON stock_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 50);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_news_embeddings_vector
        ON news_embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 50);
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_embeddings (symbol);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio_holdings (user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history (user_id, created_at DESC);")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ PostgreSQL tables created successfully")


# ── Vector search helpers ──────────────────────────────────────
def store_embedding(symbol: str, data_text: str,
                    embedding: list, metadata: dict = None):
    """Store a stock data embedding. Replaces collection.add()"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO stock_embeddings (symbol, data_text, embedding, metadata)
        VALUES (%s, %s, %s, %s)
    """, (symbol, data_text, embedding, metadata or {}))
    conn.commit()
    cur.close()
    conn.close()


def search_similar(query_embedding: list, symbol: str = None,
                   limit: int = 5) -> list:
    """Find similar stock data. Replaces collection.query()"""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    if symbol:
        cur.execute("""
            SELECT symbol, data_text, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM stock_embeddings
            WHERE symbol = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, symbol, query_embedding, limit))
    else:
        cur.execute("""
            SELECT symbol, data_text, metadata,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM stock_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results