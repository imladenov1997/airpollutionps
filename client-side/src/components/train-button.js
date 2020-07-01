import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { Button } from 'react-bootstrap';

/**
 * Train Button that is used to navigate through containers, used in the navigation bar mainly
 */
class TrainButton extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            url: '/train/'
        };
    }

    /**
     * Function that makes request to the server-side to start training a model
     */
    async trainModel() {
        let url = this.state.url + this.props.selectedModel;
        let startDate = this.props.range.startDate.format(this.props.datetimeFormat);
        let endDate = this.props.range.endDate.format(this.props.datetimeFormat);
        let locations = this.props.selectedLocations.map(coordinatesStr => this.props.getCoordinatesFromKey(coordinatesStr));
        let pollutant = this.props.pollutant;

        let body = {
            range: {
                start: startDate,
                end: endDate
            },
            data: {
                weather: this.props.weather,
            },
            locations: locations,
            pollutant: pollutant
        };
        
        try {
            await this.props.request.post(url, body);
        } catch (e) {
            return e;
        }
    }

    /**
     * Function for determining if a model can be trained depending on whether required data is input
     * @return {boolean}
     */
    isEnabled() {
        return this.props.range.startDate && this.props.range.endDate && Array.isArray(this.props.selectedLocations) && this.props.selectedLocations.length >= 2 && this.props.pollutant && this.props.selectedModel;
    }

    render() {
        return (
            <Button disabled={!this.isEnabled()} className={this.props.styles.modelButtons} variant='success' onClick={e => this.trainModel()}>
                Train Model
            </Button>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
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

export default withRouter(connect(mapStateToProps)(TrainButton));