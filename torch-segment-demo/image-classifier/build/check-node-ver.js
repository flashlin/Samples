const semver = require('semver');

if (!semver.satisfies(process.version, '>=18.13.0')) {
  console.error('Please update node version above 18.13.0');
  process.exit(1);
}