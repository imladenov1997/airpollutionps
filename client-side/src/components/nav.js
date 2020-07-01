import React, { Component } from 'react';
import { Nav, Navbar, NavItem } from 'react-bootstrap';
import RouterButton from './router-button.js';
import { connect } from 'react-redux';

/**
 * Component responsible for navigation buttons
 */
class NavigationBar extends Component {
    componentWillMount() {

    }

    componentDidMount() {

    }

    render() {
        return (
            <Navbar>
                <Nav>
                    <NavItem>
                        <RouterButton to='/' page='Home' />
                    </NavItem>
                    <NavItem>
                        <RouterButton to='/create-model' page='Create Model'/>
                    </NavItem>
                </Nav>
            </Navbar>
        )
    }
}

const mapStateToProps = state => {
    return {}
}

export default connect(mapStateToProps)(NavigationBar);