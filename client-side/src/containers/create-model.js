import React, { Component } from 'react';
import { connect } from 'react-redux';
import { withRouter } from 'react-router-dom';
import NavigationBar from '../components/nav.js';
import { Container, Row, Col, Form, Button } from 'react-bootstrap';
import DateTimeRangeComponent from '../components/date-time-range.js';
import PollutantList from '../components/pollutant-list.js';
import MapComponent from '../components/google-maps-component.js';
import ModelTypeDropdown from '../components/model-types.js';

/**
 * Class for Create Model page where a user can create a new model
 */
class CreateModel extends Component {
    constructor(props) {
        super(props);
        this.state = {
            modelName: '',
            extraData: false
        }
    }

    // Before loading page, get all types of pollution and coordinates
    async componentWillMount() {
        // Try to get all types of pollution if they are not already fetched
        if (this.props.pollutant === undefined) {
            try {
                const pollutantsUrl = '/get-pollutants';
                let pollutants = await this.props.request.get(pollutantsUrl);
                if ('success' in pollutants) {
                    this.props.dispatch({type: 'GET_ALL_POLLUTANTS', allPollutants: pollutants.success});
                }
            } catch(e) {
                return e;
            }
        }

        // Try to get all coordinates pairs if they are not already fetched
        if (this.props.locations === undefined) {
            try {
                const coordinatesUrl = '/get-coordinates';
                let coordinates = await this.props.request.get(coordinatesUrl);
                if ('success' in coordinates) {
                    this.props.dispatch({type: 'GET_ALL_LOCS', locations: coordinates.success});
                }
            } catch(e) {
                return e;
            }
        }

        this.props.dispatch({type: 'UPDATE_INITIAL_LOCATION', initialLocation: this.props.initialLocation});
    }

    componentDidMount() {

    }

    /**
     * Function for creating a model, bound to the button 'Create'
     * @param {object} e - event
     */
    async createModel(e) {
        const modelName = this.state.modelName;
        const type = this.props.selectedType; // type of the model
        const range = this.props.range; // datetime range
        const locations = this.props.selectedLocations.map(this.props.getCoordinatesFromKey);
        const pollutant = this.props.pollutant;

        const createModelUrl = '/create-model/' + modelName;
        let body = {
            type: type,
            range: {
                start: range.startDate.format(this.props.datetimeFormat),
                end: range.endDate.format(this.props.datetimeFormat)
            },
            locations: locations,
            pollutant: pollutant
        };

        if (this.state.extraData) {
            body.data = {
                weather: this.props.weather
            }
        }

        try {
            await this.props.request.post(createModelUrl, body);
        } catch(e) {
            return e;
        }
    }

    setExtraData(e) {
        this.setState({
            ...this.state,
            extraData: !this.state.extraData
        })
    }

    render() {
        return (
            <Container>
                <Row>
                    <NavigationBar />
                </Row>
                <Row>
                    <Col>
                        <Form>
                            <Form.Group>
                                <Form.Label>Model Name</Form.Label>
                                <Form.Control onChange={(e) => this.setState({modelName: e.target.value})}/>
                            </Form.Group>
                        </Form>
                        <ModelTypeDropdown />
                        <DateTimeRangeComponent />
                        <PollutantList />
                        <Form.Check
                            custom
                            inline
                            label='Add extra data'
                            type='checkbox'
                            id='custom-inline-checkbox'
                            onClick={e => this.setExtraData(e)}
                        />
                        <Button block onClick={e => this.createModel(e)}>
                            Submit
                        </Button>
                    </Col>
                    <Col md={8} sm={10} xs={10}>
                        <div style={{height: "400px"}}>
                            <MapComponent />
                        </div>
                    </Col>
                </Row>
            </Container>
        )
    }
}

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        locations: state.locations.allLocations,
        initialLocation: state.locations.initialLocation,
        range: state.datetime.range,
        request: state.initial.request,
        pollutant: state.pollutant.pollutant,
        allPollutants: state.pollutant.allPollutants,
        weather: state.weather,
        datetimeFormat: state.initial.datetimeFormat,
        selectedType: state.model.selectedType,
        selectedLocations: state.locations.selectedLocations,
        getCoordinatesFromKey: state.locations.getCoordinatesFromKey
    }
}

export default withRouter(connect(mapStateToProps)(CreateModel));