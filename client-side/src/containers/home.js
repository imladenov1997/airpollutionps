import React, { Component } from 'react';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import NavigationBar from '../components/nav.js';
import MeasurementsButton from '../components/measurements-button.js';
import PollutantList from '../components/pollutant-list.js';
import MapComponent from '../components/google-maps-component.js';
import CoordinatesList from '../components/coordinates-list.js';
import MakePredictionsComponent from '../components/make-predictions-component.js';
import ModelsList from '../components/models-list.js';
import PollutionInfo from '../components/pollution-info.js';
import DateSlider from '../components/date-slider.js';
import DateTimeRangeComponent from '../components/date-time-range.js';
import { Container, Row, Col, ButtonGroup } from 'react-bootstrap';
import TrainButton from '../components/train-button.js';

/**
 * Class for Home page
 */
class Home extends Component {
    constructor(props) {
        super(props);
        this.state = {
            chosenDatesStr: ''
        };
    }

    // Get general data from the server-side once page is about to be loaded
    async componentWillMount() {
        // Get all coordinates pairs
        let coordinates = null;
        try {
            const coordinatesUrl = '/get-coordinates';
            coordinates = await this.props.request.get(coordinatesUrl);
            if ('success' in coordinates) {
                this.props.dispatch({type: 'GET_ALL_LOCS', locations: coordinates.success});
            }
        } catch (e) {
            coordinates = [0, 0];
        }

        // Get all types of pollution
        const pollutantsUrl = '/get-pollutants';
        let pollutants = null;
        try {
            pollutants = await this.props.request.get(pollutantsUrl);
            if ('success' in pollutants) {
                this.props.dispatch({type: 'GET_ALL_POLLUTANTS', allPollutants: pollutants.success});
            }
        } catch (e) {
            pollutants = [];
        }

        // Get all models' names and types
        const modelsUrl = '/get-models';
        let modelResult = null;
        try {
            modelResult = await this.props.request.get(modelsUrl);
            if ('success' in modelResult) {
                this.props.dispatch({type: 'UPDATE_MODELS_LIST', models: modelResult.success});
            }
        } catch (e) {
            modelResult = [];
        }
    }

    componentDidMount() {

    }

    render() {
        return (
            <Container>
                <Row>
                    <NavigationBar />
                </Row>
                <Row>
                    <Col>
                        <DateTimeRangeComponent />
                        <PollutantList />
                    </Col>
                    <Col md={5}>
                        <CoordinatesList />
                    </Col>
                    <Col md={3}>
                        <MeasurementsButton />
                    </Col>
                </Row>
                <Row>
                    <Col>
                        <DateSlider />
                    </Col>
                </Row>
                <Row>
                    <Col md={8} sm={10} xs={10}>
                        <MapComponent />
                    </Col>
                    <Col md={4} sm={2} xs={2}>
                        <PollutionInfo />
                    </Col>
                </Row>
                <Row>
                    <Col md={6}>
                        <ModelsList />
                        <ButtonGroup>
                            <MakePredictionsComponent />
                            <TrainButton />
                        </ButtonGroup>
                    </Col>
                </Row>
            </Container>
        )
    }
}

// Load data from store
const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        locations: state.locations.allLocations,
        initialLocation: state.locations.initialLocation,
        range: state.datetime.range,
        request: state.initial.request,
        modelType: state.initial.modelType,
        pollutant: state.pollutant.pollutant,
        allPollutants: state.pollutant.allPollutants,
        weather: state.weather,
        datetimeFormat: state.initial.datetimeFormat,
        pollution: state.pollution.pollution
    }
}

export default withRouter(connect(mapStateToProps)(Home));