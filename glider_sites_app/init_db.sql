  
  
-- 'Metzingen':11185,
-- 'Estorf': 11001,
-- 'Leese': 10746,
-- 'Lüdingen':9759,
-- 'Brunsberg': 9844,
-- 'Kella': 9521,
-- 'Börry': 9403,
-- 'Porta': 9712,    
-- 'Königszinne': 11489,
-- 'Rammelsberg': 9427 


CREATE TABLE IF NOT EXISTS sites 
(
    site_name TEXT PRIMARY KEY, 
    dhv_site_id INT NOT NULL, 
    geo_latitude REAL,
    geo_longitude REAL
);

INSERT OR IGNORE INTO sites (site_name, dhv_site_id, geo_latitude, geo_longitude)
VALUES 
    ('Rammelsberg', 9427, 51.889874886365874, 10.43097291843072 )
;

--DROP TABLE dhv_flights;
CREATE TABLE IF NOT EXISTS dhv_flights
(
    site_name TEXT NOT NULL,
    IDFlight TEXT PRIMARY KEY,
    FlightDate TEXT NOT NULL,
    FlightStartTime TEXT NOT NULL,
    FKPilot TEXT NOT NULL,
    Glider TEXT,
    GliderClassification TEXT,
    FlightDuration REAL NOT NULL,
    BestTaskPoints REAL NOT NULL,
    BestTaskType TEXT NOT NULL,
    FOREIGN KEY(site_name) REFERENCES sites(site_name)
);