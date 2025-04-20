# language: en
Feature: Validate GenerateContext TableSchemas Functionality

  Scenario: Verify Table Schema Information
    Given I have connected to SQL Server
    Then TableSchemas should contain correct table structure information 