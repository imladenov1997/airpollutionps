import React, { Component } from 'react';
import { connect } from 'react-redux';
import Slider from 'rc-slider';
import moment  from 'moment';

/**
 * Slider component to show date and time along with slider
 */
class DateSlider extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            index: 0
        };
    }

    /**
     * Get pollution data for a specific date and time from a current location
     * @param {string} dateTime 
     */
    getPollutionInfo(dateTime) {
        let curLongitude = Array.isArray(this.props.initialLocation) ? this.props.initialLocation[0] : null;
        let curLatitude = Array.isArray(this.props.initialLocation) ? this.props.initialLocation[1] : null;
        return this.props.pollution[dateTime].filter(elem => elem.Longitude === curLongitude && elem.Latitude === curLatitude)[0];
    }

    /**
     * Show date slider
     */
    getDateSlider() {
        const createSliderWithTooltip = Slider.createSliderWithTooltip;
        const TooltipSlider = createSliderWithTooltip(Slider);

        if (typeof this.props.pollution === 'object') {
            let dateTimes = Object.keys(this.props.pollution).sort((a, b) => {
                return moment(a, this.props.datetimeFormat) - moment(b, this.props.datetimeFormat);
            });

            let NumKeys = dateTimes.length;
            return (
                <TooltipSlider 
                    min={0} 
                    max={NumKeys} 
                    tipFormatter={dateTime => dateTimes[dateTime] || 'N/A'}
                    defaultValue={this.state.index}
                    onChange={dateTime => {
                        this.props.dispatch({type: 'SELECT_DATETIME', selectedDatetime: dateTimes[dateTime]});
                        this.props.dispatch({type: 'INFO', info: this.getPollutionInfo(dateTimes[dateTime])});
                    }}
                    onAfterChange={index => this.setState({index: index})} 
                />
            );
        }

        return (
            <TooltipSlider 
                min={0} 
                max={15} disabled    
            />
        );
        
    }

    render() {
        return (
            <div style={{margin: '25px'}}>
                {this.getDateSlider()}
            </div>
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

export default connect(mapStateToProps)(DateSlider);