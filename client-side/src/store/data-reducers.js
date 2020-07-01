/**
 * Script that implements all the reducers (along with some helper functions) used across the client-side
 * Additional reducers must be implemented here
 */

 /**
  * Reducer
  * @param {object} state - Redux state
  * @param {object} action - action taken
  */
const datetimeReducer = (state = {range: {}}, action) => {
    if (action.type === 'UPDATE_DATETIME') {
        return {
            ...state,
            range: action.range
        }
    } else if (action.type === 'SELECT_DATETIME') {
        return {
            ...state,
            selectedDatetime: action.selectedDatetime
        }
    }

    return state;
};

/**
 * Turns list of coordinates pair (two entries in the list) to a string
 * @param {Array} coordinatesPair 
 * @return {string} - e.g. -1.4:51,96
 */
let getCoordinatesKey = (coordinatesPair) => {
    return coordinatesPair[0] + ':' + coordinatesPair[1]; // 0 - longitude, 1 - latitude
}

/**
 * Reverse of the above function
 * @param {string} coordinatesKey 
 * @return {Array}
 */
let getCoordinatesFromKey = (coordinatesKey) => {
    return coordinatesKey.split(':').map(numStr => parseFloat(numStr));
}

// Initial location state with default coordinates
let locationState = {
    initialLocation: [0, 0], 
    selectedLocations: [], 
    getCoordinatesKey: getCoordinatesKey,
    getCoordinatesFromKey: getCoordinatesFromKey
};

/**
 * Reducer
 * @param {object} state - Redux state
 * @param {object} action - action taken
 */
const locationReducer = (state = locationState, action) => {
    if (action.type === 'SELECT_LOCATION') {
        // Get single location
        let coordinatesStr = state.getCoordinatesKey(action.location);
        let updatedSelectedLocations = [];
        if (state.selectedLocations.includes(coordinatesStr)) {
            updatedSelectedLocations = state.selectedLocations.filter(coordinates => coordinates !== coordinatesStr); 
        } else {
            updatedSelectedLocations = state.selectedLocations.map(e => e);
            updatedSelectedLocations.push(coordinatesStr);
        }

        return {
            ...state,
            initialLocation: action.location,
            selectedLocations: updatedSelectedLocations
        };
    } else if (action.type === 'GET_ALL_LOCS') {
        // get all locations
        return {
            ...state,
            allLocations: action.locations
        };
    } else if (action.type === 'UPDATE_INITIAL_LOCATION') {
        // update default (in the beginning, initial) location
        return {
            ...state,
            initialLocation: action.initialLocation
        }
    } else if (action.type === 'ADD_NEW_LOCATION') {
        // when double clicking on a place in the map
        if (Array.isArray(state.allLocations)) {
            let allLocationsUpdated = state.allLocations.map(e => e);
            allLocationsUpdated.push(action.newLocation);
            return {
                ...state,
                allLocations: allLocationsUpdated,
                initialLocation: action.newLocation
            }
        }

        return state;
        
    }

    return state;
}

/**
 * Reducer for dealing with pollution data
 * @param {object} state - Redux state
 * @param {object} action - action taken
 */
const pollutantReducer = (state = {}, action) => {
    if (action.type === 'UPDATE_POLLUTANT') {
        // Update selected type of pollution
        return {
            ...state,
            pollutant: action.pollutant
        }
    } else if (action.type === 'GET_ALL_POLLUTANTS') {
        // Get all pollutants available
        return {
            ...state,
            allPollutants: action.allPollutants
        }
    }

    return state
}

// Some weather data that is saved in the database, currently this is hardcoded...
let weatherState = {
    Temperature: null,
    Humidity: null,
    Precipitation: null,
    WindSpeed: null
};

/**
 * Reducer, currently these reducers are not used
 * @param {*} state - Redux state
 * @param {*} action - action to update state
 */
const weatherReducer = (state = weatherState, action) => {
    if (action.type === 'ADD_METEOROLOGICAL_FACTOR') {

        let newState = {
            ...state
        };
        newState[action.factor] = null;
        return state;
    } else if (action.type === 'REMOVE_METEOROLOGICAL_FACTOR') {
        let newState = {
            ...state
        };

        delete newState[action.factor];
        return newState;
    }

    return state;
}

/**
 * Reducer
 * @param {*} state 
 * @param {*} action 
 */
const modelReducer = (state = {}, action) => {
    if (action.type === 'UPDATE_MODELS_LIST') {
        // Update list with models' names and types
        return {
            ...state,
            models: action.models
        };
    } else if (action.type === 'UPDATE_SELECTED_MODEL') {
        // Selects the chosen by the user model
        return {
            ...state,
            selectedModel: action.model
        };
    } else if (action.type === 'GET_MODEL_TYPES') {
        // Get all existing models' types that the system supports
        return {
            ...state,
            modelTypes: action.modelTypes
        };
    } else if (action.type === 'UPDATE_SELECTED_TYPE') {
        // Selects the chosen by the user model type that the system currently supports
        return {
            ...state,
            selectedType: action.selectedType
        };
    }

    return state;
}

/**
 * Reducer dealing with pollution data
 * @param {*} state 
 * @param {*} action 
 */
const pollutionReducer = (state = {info: false}, action) => {
    if (action.type === 'POLLUTION') {
        const dataset = action.result;
        let pollution = dataset.reduce((prev, item) => {
            const dateTimeKey = item.DateTime;
            if (prev[dateTimeKey] === undefined) {
                prev[dateTimeKey] = [];
            }

            delete item[dateTimeKey];
            prev[dateTimeKey].push(item);
            return prev;
        }, {});
        return {
            ...state,
            pollution: pollution
        };
    } else if (action.type === 'INFO') {
        let info = action.info;
        return {
            ...state,
            info: info
        };
    }

    return state;
}

export {datetimeReducer, locationReducer, pollutantReducer, weatherReducer, modelReducer, pollutionReducer};