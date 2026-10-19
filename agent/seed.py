"""Synthetic demo customers and rentals. All names and contacts are fictional
(example.com addresses, 555-01xx phone numbers)."""

CUSTOMERS = [
    ("C-1001", "Mara Lindqvist", "mara.lindqvist@example.com", "+1-555-0101", "plus"),
    ("C-1002", "Tomas Okafor", "tomas.okafor@example.com", "+1-555-0102", "standard"),
    ("C-1003", "Priya Nandakumar", "priya.n@example.com", "+1-555-0103", "standard"),
]

# (id, customer, bike, start offset days, end offset days, status, daily rate)
RENTALS = [
    ("R-2001", "C-1001", "City e-bike", -1, 2, "active", 32.00),
    ("R-2002", "C-1001", "Tandem bike", 7, 9, "booked", 45.00),
    ("R-2003", "C-1001", "Road bike", -40, -37, "returned", 28.00),
    ("R-2004", "C-1002", "Mountain bike", 3, 5, "booked", 35.00),
    ("R-2005", "C-1002", "Cargo bike", -12, -10, "returned", 40.00),
    ("R-2006", "C-1003", "Hybrid bike", 0, 4, "active", 24.00),
    ("R-2007", "C-1003", "Kids bike", 10, 11, "booked", 12.00),
]


def reset_demo(conn) -> int:
    """Restore customers and rentals to the seed state and clear memories and the
    action log. Works with a sync psycopg connection; returns the customer count."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories")
        cur.execute("DELETE FROM action_log")
        cur.execute("DELETE FROM rentals")
        cur.execute("DELETE FROM customers")
        cur.executemany(
            "INSERT INTO customers (id, name, email, phone, tier) VALUES (%s, %s, %s, %s, %s)",
            CUSTOMERS,
        )
        cur.executemany(
            "INSERT INTO rentals (id, customer_id, bike, start_date, end_date, status, daily_rate)"
            " VALUES (%s, %s, %s, CURRENT_DATE + %s, CURRENT_DATE + %s, %s, %s)",
            RENTALS,
        )
    conn.commit()
    return len(CUSTOMERS)
