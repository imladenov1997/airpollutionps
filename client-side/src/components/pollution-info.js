import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { Table } from 'react-bootstrap';

/**
 * Class for showing some stats on pollution for a selected value
 */
class PollutionInfo extends Component {
    /**
     * Function for getting required data from pollution information for a given location and point in time
     * @param {string} feature - feature to look into
     */
    getFeatureValue(feature) {
        let returnedValue = 'N/A';
        if (this.props.pollutionInfo) {
            switch (feature) {
                case 'DateTime': 
                    returnedValue = this.props.pollutionInfo.DateTime;
                    break;
                case 'Longitude': 
                    returnedValue = this.props.pollutionInfo.Longitude;
                    break;
                case 'Latitude': 
                    returnedValue = this.props.pollutionInfo.Latitude;
                    break;
                case 'Pollutant': 
                    returnedValue = this.props.pollutionInfo.Pollutant;
                    break;
                case 'Uncertainty': 
                    returnedValue = this.props.pollutionInfo.Uncertainty || 0; // uncertainty may be null
                    break;
                default:
                    break;
            }
                
        }       
        
        return returnedValue;
    }

    render() {
        const styles = this.props.styles.pollutionInfo;
        return (
            <Table className={styles}>
                <thead>
                    <tr>
                        <th>Feature</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Date and Time</td>
                        <td>{this.getFeatureValue('DateTime')}</td>
                    </tr>
                    <tr>
                        <td>Coordinates</td>
                        <td>Longitude: {this.getFeatureValue('Longitude')} | Latitude: {this.getFeatureValue('Latitude')}</td>
                    </tr>
                    <tr>
                        <td>Pollution Level</td>
                        <td>{this.getFeatureValue('Pollutant')}</td>
                    </tr>
                    <tr>
                        <td>Uncertainty</td>
                        <td>{this.getFeatureValue('Uncertainty')}</td>
                    </tr>
                </tbody>
            </Table>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        pollutionInfo: state.pollution.info
    }
}

export default withRouter(connect(mapStateToProps)(PollutionInfo));