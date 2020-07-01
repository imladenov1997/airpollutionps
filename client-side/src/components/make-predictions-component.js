import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { Button } from 'react-bootstrap';

/**
 * Navigation Button that is used to navigate through containers, used in the navigation bar mainly
 */
class MakePredictionsComponent extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            url: '/predict'
        };
    }

    /**
     * Make POST request for a model to make predictions
     */
    async makePredictions() {
        let startDate = this.props.range.startDate.format(this.props.datetimeFormat);
        let endDate = this.props.range.endDate.format(this.props.datetimeFormat);
        let locations = this.props.selectedLocations.map(coordinatesStr => this.props.getCoordinatesFromKey(coordinatesStr));
        let selectedModel = this.props.selectedModel;

        let body = {
            name: selectedModel,
            range: {
                start: startDate,
                end: endDate
            },
            locations: locations,
            pollutant: this.props.pollutant,
            data: {
                weather: this.props.weather
            },
            uncertainty: true
        };
        
        try {
            await this.props.request.post(this.state.url, body);
        } catch (e) {
            return e;
        }
    }

    /**
     * Enable the button to be clickable so the user can send the request once necessary data is input
     */
    isEnabled() {
        return this.props.range.startDate && this.props.range.endDate && Array.isArray(this.props.selectedLocations) && this.props.selectedLocations.length >= 2 && this.props.pollutant && this.props.selectedModel;
    }

    render() {
        return (
            <Button disabled={!this.isEnabled()} className={this.props.styles.modelButtons} variant='success' onClick={e => this.makePredictions()}>
                Make Predictions
            </Button>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        locations: state.locations.allLocations,
        initialLocation: state.locations.initialLocation,
        selectedLocations: state.locations.selectedLocations,
        getCoordinatesFromKey: state.locations.getCoordinatesFromKey,
        range: state.datetime.range,
        request: state.initial.request,
        modelType: state.initial.modelType,
        pollutant: state.pollutant.pollutant,
        weather: state.weather,
        datetimeFormat: state.initial.datetimeFormat,
        selectedModel: state.model.selectedModel
    }
}

export default withRouter(connect(mapStateToProps)(MakePredictionsComponent));