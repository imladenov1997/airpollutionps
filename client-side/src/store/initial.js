import request from '../api/api.js';
import styles from '../style/styles.css';

// Datetetime format that will be used in the client-side, kept consistent with the one on server-side
const datetimeFormat = 'DD-MM-YYYY HH:mm';

let config = {
    AirPollution: process.env.REACT_APP_AirPollution,
    GoogleApiKey: process.env.REACT_APP_GoogleApiKey,
    GoogleBluePinURL: process.env.REACT_APP_GoogleBluePinURL
};

// Redux intial state
const initialState = {
    request: request(process.env.REACT_APP_AirPollution),
    styles: styles,
    config: config,
    datetimeFormat: datetimeFormat
};

const createInitialState = (state = initialState, action) => state;

export {createInitialState};