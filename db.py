import psycopg2
from psycopg2.extras import RealDictCursor
from config import settings

DATABASE_URL = settings['database_url']

# ── Connect to PostgreSQL ──────────────────────────────────────
def get_connection(register_vec=False):
    """Get a raw psycopg2 connection. Always close after use."""
    conn = psycopg2.connect(DATABASE_URL)
    if register_vec:
        try:
            from pgvector.psycopg2 import register_vector
            register_vector(conn)
        except Exception:
            pass  # pgvector not available - skip
    return conn

def get_cursor(conn):
    """Get a cursor that returns rows as dicts."""
    return conn.cursor(cursor_factory=RealDictCursor)

# ── Create all tables on first run ────────────────────────────
def init_db():
    conn = get_connection(register_vec=False)
    cur  = conn.cursor()

    # Try to enable pgvector - skip if not available
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        has_vector = True
    except Exception:
        conn.rollback()
        has_vector = False

    # Enable uuid-ossp
    try:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        conn.commit()
    except Exception:
        conn.rollback()

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

    # Agent memory (without vector if pgvector unavailable)
    if has_vector:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id      UUID REFERENCES users(id) ON DELETE CASCADE,
                memory_type  TEXT DEFAULT 'fact',
                content      TEXT NOT NULL,
                embedding    vector(1536),
                created_at   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id      UUID REFERENCES users(id) ON DELETE CASCADE,
                memory_type  TEXT DEFAULT 'fact',
                content      TEXT NOT NULL,
                created_at   TIMESTAMPTZ DEFAULT NOW()
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

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_user ON portfolio_holdings (user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history (user_id, created_at DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_user ON agent_memory (user_id);")

    conn.commit()
    cur.close()
    conn.close()
    print("PostgreSQL tables ready (pgvector: {})".format("enabled" if has_vector else "skipped"))