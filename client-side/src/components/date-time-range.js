import React, { Component } from 'react';
import { connect } from 'react-redux';
import DatetimeRangePicker from 'react-bootstrap-datetimerangepicker';
import { Button, InputGroup, FormControl } from 'react-bootstrap';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCalendar } from '@fortawesome/free-solid-svg-icons';

/**
 * Component for date and time range
 */
class DateTimeRangeComponent extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            chosenDatesStr: ''
        };
    }

    getDateAndTime(picker) {
        this.setState({
            startDate: picker.startDate,
            endDate: picker.endDate,
            chosenDatesStr: picker.startDate.format(this.props.datetimeFormat) + ' - ' + picker.endDate.format(this.props.datetimeFormat)
        });

        let range = {
            startDate: this.state.startDate,
            endDate: this.state.endDate
        };

        this.props.dispatch({type: 'UPDATE_DATETIME', range: range});
    }

    render() {
        return (
            <DatetimeRangePicker
                className={this.props.styles.dateTimePicker}
                timePicker
                timePicker24Hour
                showDropdowns
                timePickerSeconds
                startDate={this.state.startDate}
                endDate={this.state.endDate}
                onApply={(_, picker) => this.getDateAndTime(picker)}>
                
                <InputGroup className="mb-3">
                    <FormControl
                        placeholder='Pick a date range'
                        defaultValue={this.state.chosenDatesStr}
                    />
                    <InputGroup.Append>
                    <Button variant='outline-dark'>
                        <FontAwesomeIcon icon={faCalendar} ></FontAwesomeIcon>
                    </Button>
                    </InputGroup.Append>
                </InputGroup>
            </DatetimeRangePicker>
        );
    }
  }

const mapStateToProps = state => {
    return {
        pollution: state.pollution.pollution,
        initialLocation: state.locations.initialLocation,
        styles: state.initial.styles,
        datetimeFormat: state.initial.datetimeFormat
    }
}

export default connect(mapStateToProps)(DateTimeRangeComponent);