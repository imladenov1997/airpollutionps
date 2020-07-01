import React, { Component } from 'react';
import { withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { ListGroup } from 'react-bootstrap';

/**
 * Component for showing a list of all model's names and their respective types
 */
class ModelsList extends Component {
    constructor(props) {
        super(props);
        this.state = { 
            url: '/get-models'
        };
    }

    /**
     * Function for selecting a model and updating the store
     * @param {string} modelName 
     */
    makeActive(modelName) {
        this.props.dispatch({type: 'UPDATE_SELECTED_MODEL', model: modelName});
    }

    /**
     * Function for determining the color of the button once a model is selected
     * @param {string} modelName 
     */
    getButtonColour(modelName) {
        return this.props.selectedModel === modelName ? 'primary' : null;
    }

    /**
     * Function for generating the list of existing models
     */
    generateModelList() {
        let modelList = this.props.models;
        if (Array.isArray(modelList)) {
            let modelElements = modelList.map((model, index) => {
                return (
                    <ListGroup.Item variant={this.getButtonColour(model.name)} action key={index+1} onClick={e => this.makeActive(model.name)}>
                        Model: {model.name} | Type: {model.type}
                    </ListGroup.Item>
                )
            });
            return modelElements;
        }

        return [];
    }

    render() {
        const listGroupStyle = this.props.styles.modelsList;
        return (
            <ListGroup className={listGroupStyle} defaultActiveKey="#link1">
                <ListGroup.Item variant='light' key={0}>
                    Models
                </ListGroup.Item>
                {this.generateModelList()}
            </ListGroup>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles,
        locations: state.locations,
        request: state.initial.request,
        models: state.model.models,
        selectedModel: state.model.selectedModel
    }
}

export default withRouter(connect(mapStateToProps)(ModelsList));