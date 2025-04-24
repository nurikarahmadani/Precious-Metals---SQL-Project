## Database Preparation
```sql
CREATE DATABASE precious_metals

CREATE TABLE gold(
	gold_timestamp DATE,
	gold_open DECIMAL,
	gold_high DECIMAL,
	gold_low DECIMAL,
	gold_close DECIMAL,
	volume DECIMAL,
	currency VARCHAR(20),
	unit VARCHAR(20),
	headlines VARCHAR(100)
)
```