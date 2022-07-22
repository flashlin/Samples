import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
    stages: [
        { duration: '30s', target: 10 },
        { duration: '30s', target: 30 },
        { duration: '30s', target: 100 },
        { duration: '30s', target: 200 },
    ],
};

export default function () {
   let baseUrl = 'http://172.23.144.1:5063';
   let res = http.get(`${baseUrl}/AsyncInView`);
   check(res, { 'status was 200': (r) => r.status == 200 });
   sleep(1);
}