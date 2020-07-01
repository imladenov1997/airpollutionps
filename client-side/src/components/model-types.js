import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { DropdownButton, Dropdown } from 'react-bootstrap';

/**
 * Dropdown button component for choosing model type
 */
class ModelTypeDropdown extends Component {
    constructor(props) {
        super(props);
        this.state = {
            modelTypes: ['']
        }
    }

    /**
     * Get existing model types from server-side
     */
    async componentWillMount() {
        const modelTypesUrl = '/get-model-types';
        let modelTypesResult = null;
        try {
            modelTypesResult = await this.props.request.get(modelTypesUrl);
            if ('success' in modelTypesResult) {
                this.setState({
                    modelTypes: modelTypesResult.success
                });
                if (Array.isArray(this.state.modelTypes)) {
                    let initial = this.state.modelTypes[0];
                    this.updateSelected(initial);
                }
            }
        } catch(e) {
            modelTypesResult = [];
        }
    }

    /**
     * Update store with selected model types
     * @param {string} type 
     */
    updateSelected(type) {
        this.props.dispatch({type: 'UPDATE_SELECTED_TYPE', selectedType: type});
    }

    render() {
        let options = null; 
        
        if (Array.isArray(this.state.modelTypes)) {
            options = this.state.modelTypes.map((type, index) => {
                return (
                    <Dropdown.Item key={index} eventKey={type} onSelect={(eventKey) => this.updateSelected(eventKey)}>
                        {type}
                    </Dropdown.Item>
                )
            });
        }

        return (
            <DropdownButton title={this.props.selectedType || 'Select a Model'}>
                {options}
            </DropdownButton>
        );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        modelTypeList: state.model.modelTypes,
        request: state.initial.request,
        selectedType: state.model.selectedType
    }
}

export default withRouter(connect(mapStateToProps)(ModelTypeDropdown));