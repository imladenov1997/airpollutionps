import React, { Component } from 'react';
import { Redirect, withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import { Button } from 'react-bootstrap';

/**
 * Router Button that is used to navigate through containers, used in the navigation bar mainly
 */
class RouterButton extends Component {
    constructor(props) {
        super(props);
        this.state = { redirect: false };
    }

    // Redirect to a new container
    click = () => {
        this.props.history.push(this.props.to);
    }

    render() {
        if (this.state.redirect) {
            return <Redirect to={this.props.to} />
        }

        return (
            <Button className={this.props.styles.routerButton} variant='outline-primary' onClick={this.click}>
                {this.props.page}
            </Button>
      );
    }
  }

const mapStateToProps = state => {
    return {
        styles: state.initial.styles
    }
}

export default withRouter(connect(mapStateToProps)(RouterButton));