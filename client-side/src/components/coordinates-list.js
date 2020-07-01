import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { ListGroup } from 'react-bootstrap';

/**
 * Class for keeping the list of coordinate pairs and show them in a user-friendly way
 */
class CoordinatesList extends Component {
    /**
     * Select a coordinates pair
     */
    makeActive(coordinatesPair) {
        this.props.dispatch({type: 'UPDATE_INITIAL_LOCATION', initialLocation: coordinatesPair});
    }

    /**
     * Depending on whether a coordinates pair is chosen, a button is either blue or white
     * @param {Array} rowCoordinates 
     * @return {string}
     */
    getButtonColour(rowCoordinates) {
        if (Array.isArray(this.props.initialLocation) && Array.isArray(rowCoordinates)) {
            return this.props.initialLocation[0] === rowCoordinates[0] && 
                   this.props.initialLocation[1] === rowCoordinates[1] ? 'primary' : null;
        }

        return null;
    }

    /**
     * Label to be shown on each button
     * @param {Array} coordinatesPair 
     * @return {string} parsedCoordinates
     */
    coordinatesToText(coordinatesPair) {
        let parsedCoordinates = "";
        if (Array.isArray(coordinatesPair) && coordinatesPair.length) {
            parsedCoordinates = "Longitude: " + coordinatesPair[0] + ", ";
            parsedCoordinates += "Latitude: " + coordinatesPair[1] + ", ";
        }

        return parsedCoordinates;
    }

    /**
     * Create the coordinates list that is shown to the user
     */
    generateCoordinateList() {
        let coordinates = this.props.locations.allLocations;
        let coordinatesElements = [];
        if (Array.isArray(coordinates)) {
            coordinatesElements = coordinates.map((coordinatesPair, index) => {
                return (
                    <ListGroup.Item variant={this.getButtonColour(coordinatesPair)} action key={index} onClick={e => this.makeActive(coordinatesPair)}>
                        {this.coordinatesToText(coordinatesPair)}
                    </ListGroup.Item>
                )
            });
        }
        return coordinatesElements;
    }

    render() {
        const style = this.props.styles.coordinatesList;
        return (
            <ListGroup className={style} defaultActiveKey="#link1">
                <ListGroup.Item variant='light'>
                    Coordinates
                </ListGroup.Item>
                {this.generateCoordinateList()}
            </ListGroup>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        locations: state.locations,
        initialLocation: state.locations.initialLocation
    }
}

export default withRouter(connect(mapStateToProps)(CoordinatesList));