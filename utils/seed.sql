\connect postgres

DROP DATABASE IF EXISTS political_party_classifier;
CREATE DATABASE political_party_classifier;

\connect political_party_classifier

CREATE TABLE convote_dataset (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    party VARCHAR(255) NOT NULL,
    party_num_label INT NOT NULL
);

CREATE TABLE convote_predictions (
     id SERIAL PRIMARY KEY,
     text TEXT NOT NULL,
     party VARCHAR(255) NOT NULL,
     party_num_label INT NOT NULL
);
