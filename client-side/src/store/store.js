// Initial state for redux
import { combineReducers, createStore } from 'redux';
import { createInitialState } from './initial.js';
import { datetimeReducer, locationReducer, pollutantReducer, weatherReducer, modelReducer, pollutionReducer } from './data-reducers.js';

/**
 * This is the main store for redux
 */
let combinedReducers = combineReducers({
    initial: createInitialState,
    datetime: datetimeReducer,
    locations: locationReducer,
    pollutant: pollutantReducer,
    weather: weatherReducer,
    model: modelReducer,
    pollution: pollutionReducer
});

export default createStore(combinedReducers);