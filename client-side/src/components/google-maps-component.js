import {GoogleApiWrapper, Map, Marker, Circle} from 'google-maps-react';
import React, { Component } from 'react';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';

const MAX_OPACITY = 0.8;

/**
 * Component that wraps around Google Maps
 */
export class MapComponent extends Component {
    constructor(props) {
        super(props);
        this.state = {
            mapCenter: {
                lng: this.props.locations.initialLocation[0],
                lat: this.props.locations.initialLocation[1]
            },
            routes: {
                singlePrediction: '/predict-single-instance'
            }
        };
    }

    /**
     * Once double-clicked on a location, if existing data is selected, a prediction for this location for given time is made
     * It is important to note that this prediction does not take into consideration meteorological factors
     * @param {object} event - clicked event
     */
    async createNewLocation(event) {
        // Make predictions only when at home page
        if (this.props.location.pathname !== '/') {
            return;
        }

        // Get coordinates
        let longitude = parseFloat(event.latLng.lng().toFixed(8));
        let latitude = parseFloat(event.latLng.lat().toFixed(8));
        let coordinates = [longitude, latitude];

        // POST body
        let body = {
            name: this.props.model.selectedModel, // TODO - to be changed
            date_time: this.props.datetime.selectedDatetime,
            longitude: longitude,
            latitude: latitude,
            pollutant: this.props.pollutant.pollutant, // TODO - to be changed
            uncertainty: true
        };

        try {
            let result = await this.props.request.post(this.state.routes.singlePrediction, body);
            if ('success' in result) {
                this.props.dispatch({type: 'ADD_NEW_LOCATION', newLocation: coordinates});
            }
        } catch (e) {
            return e;
        }
    }

    /**
     * Check if coordinates pair is selected
     * @param {Array} coordinates
     */
    isCoordinatesPairSelected(coordinates) {
        let key = this.props.locations.getCoordinatesKey(coordinates);
        return this.props.locations.selectedLocations.includes(key);
    }

    /**
     * Depending on whether a model is selected or not, get red or blue pin (blue for selected)
     * @param {*} coordinates 
     */
    getRedOrBluePin(coordinates) {
        let pin = null;
        if (this.isCoordinatesPairSelected(coordinates)) {
            pin = {
                url: this.props.bluePinURL
            };
        }

        return pin;
    }

    selectLocation(event) {
        let coordinates = [
            event.position.lng,
            event.position.lat
        ];

        this.props.dispatch({type: 'SELECT_LOCATION', location: coordinates});
    }

    getCoordinatesPair(dataInstance) {
        let coordinatesPair = [];
        coordinatesPair.push(dataInstance.Longitude);
        coordinatesPair.push(dataInstance.Latitude);
        return coordinatesPair;
    }

    addPollutionCircle(elem, index) {
        let colour;
        let coordinates = this.getCoordinatesPair(elem);
        let pollutionLevel = elem.Pollutant;
        let uncertainty = elem.Uncertainty;
        if (pollutionLevel < 10) {
            colour = '#009933';
        } else if (pollutionLevel < 25) {
            colour = '#FFFF00';
        } else if (pollutionLevel < 40) {
            colour = '#FF9933';
        } else {
            colour = '#FF0000';
        }

        let opacity = 0
        if (uncertainty === 0) {
            opacity = MAX_OPACITY;
        } else {
            opacity = MAX_OPACITY/(1 + uncertainty); // 1 + uncertainty to avoid values between 0 and 1 as they will increase opacity
        }

        return (
            <Circle
                key={index}
                radius={400}
                center={{lng: coordinates[0], lat: coordinates[1]}}
                onClick={() => {
                    this.props.dispatch({type: 'UPDATE_INITIAL_LOCATION', initialLocation: coordinates});
                    this.props.dispatch({type: 'INFO', info: elem});
                }}
                strokeColor='transparent'
                strokeOpacity={0}
                strokeWeight={10}
                fillColor={colour}
                fillOpacity={opacity}   
            />
        );
    }

    addPollutionCircles(dataset, dateTime) {
        if (typeof dataset === 'object') {
            let pollutionLevels = dataset[dateTime];
            if (Array.isArray(pollutionLevels)) {
                let existingPollutionLevels = pollutionLevels.filter(elem => elem.Pollutant !== null);
                return existingPollutionLevels.map((elem, index) => {
                    return this.addPollutionCircle(elem, index);
                });
            }
        }

        return [];
    }

    addPollutionMarker(coordinatesPair, index) {
        return (
            <Marker icon={this.getRedOrBluePin(coordinatesPair)} key={index} onClick={(event) => this.selectLocation(event)} position={{lng: coordinatesPair[0], lat: coordinatesPair[1]}} />
        );
    }

    addPollutionMarkers(coordinates) {
        return Array.isArray(coordinates) ? coordinates.map((elem, index) => this.addPollutionMarker(elem, index)) : [];
    }

    render() {
        let mapCenter = {
            lng: this.props.locations.initialLocation[0],
            lat: this.props.locations.initialLocation[1]
        };

        return (
            <div>
                <Map google={this.props.google}
                     zoom={14}
                     center={mapCenter}
                     onDblclick={(event, eventTwo, eventThree) => this.createNewLocation(eventThree)}
                     >
                    {this.addPollutionCircles(this.props.pollution.pollution, this.props.datetime.selectedDatetime)}
                    {this.addPollutionMarkers(this.props.locations.allLocations)}
                </Map>
            </div>
        );
    }
};

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        datetime: state.datetime,
        locations: state.locations,
        pollution: state.pollution,
        dateTimeFormat: state.initial.datetimeFormat,
        bluePinURL: state.initial.config.GoogleBluePinURL,
        request: state.initial.request,
        model: state.model,
        pollutant: state.pollutant
    }
}

export default GoogleApiWrapper({
    apiKey: (process.env.REACT_APP_GoogleApiKey),
    libraries: ['visualization']
})(withRouter(connect(mapStateToProps)(MapComponent)));