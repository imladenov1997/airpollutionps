import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { ListGroup} from 'react-bootstrap';

/**
 * Component that lists all types of pollution
 */
class PollutantList extends Component {
    /**
     * Highlight a selected pollutant and update the value in the store
     * @param {string} name 
     */
    makeActive(name) {
        this.props.dispatch({type: 'UPDATE_POLLUTANT', pollutant: name});
    }

    /**
     * Show all pollutants in the list
     */
    generatePollutantItems() {
        let pollutants = this.props.allPollutants;
        let pollutantElements = [];
        
        if (Array.isArray(pollutants)) {
            pollutantElements = pollutants.map((pollutant, index) => {
                return (
                    <ListGroup.Item variant={pollutant === this.props.selectedPollutant ? 'primary' : null} action key={index+1} onClick={e => this.makeActive(pollutant)}>
                        {pollutant}
                    </ListGroup.Item>
                )
            });
        }

        return pollutantElements;
    }

    render() {
        const listGroupStyle = this.props.styles.pollutionList;
        return (
            <ListGroup className={listGroupStyle} defaultActiveKey="#link1">
                <ListGroup.Item variant='light' key={0}>
                    Pollutant
                </ListGroup.Item>
                {this.generatePollutantItems()}
            </ListGroup>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        allPollutants: state.pollutant.allPollutants,
        selectedPollutant: state.pollutant.pollutant
    }
}

export default withRouter(connect(mapStateToProps)(PollutantList));