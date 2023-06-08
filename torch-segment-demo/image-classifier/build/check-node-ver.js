const semver = require('semver');

const currentVersion = process.version;
console.log(`current node ver is ${currentVersion}`);
if (!semver.satisfies(process.version, '>=16.13.1')) {
  console.error('Please update node version above 16.13.1');
  process.exit(1);
}