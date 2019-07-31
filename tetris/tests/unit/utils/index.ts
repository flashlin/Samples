import withMessage from './AssertionWithMessage';
global.expect = withMessage(global.expect);