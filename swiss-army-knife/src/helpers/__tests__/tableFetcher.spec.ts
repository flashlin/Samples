import { describe, it, expect } from 'vitest'
//import { flushPromises, mount, VueWrapper } from '@vue/test-utils'
import { fetchAllTable, fetchTableData } from "../tableFetcher"

import data_table_json_expected from "./data/datatable_expected.json";
import { readFileContent } from './testHelper';
describe('fetchTableData', () => {
    //let app: VueWrapper
    //beforeEach(() => {
    // app = mount(LoginName, {
    //     global: {
    //         mocks: {
    //             $t: vi.fn(),
    //         },
    //     },
    // })
    //})

    //const loginNameInput = app.findByTestId('loginName')
    //await loginNameInput.setValue('test')
    //app.findByTestId('loginNameTips')
    //await flushPromises()
    //expect(getTipListElement().exists()).toBeTruthy()

    it('table thead tr td', () => {
        const fakeTable = document.createElement('table');
        fakeTable.innerHTML = readFileContent("data/thead_tr_td.xml");
        const dataTable = fetchTableData(fakeTable);
        expect(dataTable).toStrictEqual(data_table_json_expected)
    })

    it('table tr th', () => {
        const fakeTable = document.createElement('table');
        fakeTable.innerHTML = readFileContent("data/table_tr_th.xml");
        const dataTable = fetchTableData(fakeTable);
        expect(dataTable).toStrictEqual(data_table_json_expected)
    })


    it('table', async () => {
        document.body.innerHTML = readFileContent("data/thead_tr_td.xml");
        const dataTableList = fetchAllTable();
        console.log(dataTableList);
    })

})
