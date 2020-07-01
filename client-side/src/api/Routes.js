import React, { Component } from 'react';
import { Route, Switch } from 'react-router-dom';
import Home from '../containers/home.js';
import CreateModel from '../containers/create-model.js';

/**
 * Component for defining all the routes within the application and the containers they are mapped to
 */
class Routes extends Component {
    render() {
        return (
            <Switch>
                <Route exact path='/' component={Home} />
                <Route exact path='/create-model' component={CreateModel} />
            </Switch>
        )
    }
}

export default Routes;