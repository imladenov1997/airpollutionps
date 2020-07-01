import React, { Component } from 'react';
import Routes  from './api/Routes.js';
import { BrowserRouter } from 'react-router-dom';
import './style/App.css';
import { connect } from 'react-redux';

/**
 * Base class of the single-page app
 */
class App extends Component {
  render() {
    return (
      <BrowserRouter>
        <Routes />
      </BrowserRouter>
    );
  }
}

/**
 * 
 * @param {object} state - Redux object/state
 * @return {object} state - updated Redux state
 */
const mapStateToProps = state => {
  return {}
}

export default connect(mapStateToProps)(App);

