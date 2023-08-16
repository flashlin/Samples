import './assets/main.css';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import router from './router';

import './db';
import { parseTsql } from './sqlex/tsql';
//import { Tsql } from './antlr/TSQL';
//const tsql = new Tsql();
parseTsql('select id from customer');

import { contentDiv } from "./init"
const app = createApp(App);
app.use(createPinia());
app.use(router);
//app.mount('#app');
app.mount(contentDiv);


/*
<table class="DT" id="resultTable">
        <thead>
            <tr>
                <td class="W30" id="rowItemHeader">#</td>
                            <td>ISOCurrency</td>
                            <td>AgentName</td>
                            <td>SubAgentName</td>
                            <td>Enable</td>
                            <td>ModifyBy</td>
                            <td>ModifyOn</td>

            </tr>
        </thead>
        <tbody id="rowItem">
                    <tr class="TrOdd">
                        <td class="W30 TAC">1</td>
                            <td>IDR</td>
                            <td>sbotb</td>
                            <td>psrp</td>
                            <td>True</td>
                            <td>tleon</td>
                            <td>2020-02-18 03:52:53.650</td>
                    </tr>
                    <tr class="TrEven">
                        <td class="W30 TAC">2</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbksk01</td>
                            <td>False</td>
                            <td>Vincent Lai</td>
                            <td>2020-12-17 04:37:55.567</td>
                    </tr>
                    <tr class="TrOdd">
                        <td class="W30 TAC">7</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbbbl01</td>
                            <td>False</td>
                            <td>Vincent Lai</td>
                            <td>2020-12-17 04:37:55.567</td>
                    </tr>
                    <tr class="TrEven">
                        <td class="W30 TAC">8</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbksk02</td>
                            <td>True</td>
                            <td>Jason Chen</td>
                            <td>2020-12-06 23:18:17.003</td>
                    </tr>
                    <tr class="TrOdd">
                        <td class="W30 TAC">9</td>
                            <td>THB</td>
                            <td>paytb</td>
                            <td>paytbtmb02</td>
                            <td>True</td>
                            <td>Jason Chen</td>
                            <td>2020-12-06 23:18:17.003</td>
                    </tr>
        </tbody>
    </table>
*/