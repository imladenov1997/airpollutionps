import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { Button } from 'react-bootstrap';

/**
 * Navigation Button that is used to navigate through containers, used in the navigation bar mainly
 */
class MeasurementsButton extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            url: '/get-pollution'
        };
    }

    /**
     * Get pollution levels for a given set of locations and a selected range and update the store
     */
    async getMeasurements() {
        let startDate = this.props.range.startDate.format(this.props.datetimeFormat);
        let endDate = this.props.range.endDate.format(this.props.datetimeFormat);
        let locations = this.props.locations;

        let body = {
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
            let result = await this.props.request.post(this.state.url, body);
            if ('success' in result) {
                this.props.dispatch({type: 'POLLUTION', result: result.success});
            } 
        } catch (e) {
            return e;
        }
    }

    /**
     * Enable 'Get Measurements' button once required data is input
     */
    isEnabled() {
        return this.props.range.startDate && this.props.range.endDate && this.props.locations && this.props.pollutant;
    }

    render() {
        return (
            <Button disabled={!this.isEnabled()} block className={this.props.styles.routerButton} variant='success' onClick={e => this.getMeasurements()}>
                Get Measurements
            </Button>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        locations: state.locations.allLocations,
        range: state.datetime.range,
        request: state.initial.request,
        modelType: state.initial.modelType,
        pollutant: state.pollutant.pollutant,
        weather: state.weather,
        datetimeFormat: state.initial.datetimeFormat
    }
}

export default withRouter(connect(mapStateToProps)(MeasurementsButton));