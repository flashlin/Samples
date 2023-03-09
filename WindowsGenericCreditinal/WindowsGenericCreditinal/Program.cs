using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.Net;
using CredentialManagement;

namespace WindowsGenericCreditinal
{
    class Program
    {
        static void Main(string[] args)
        {
            var credentialSet = new CredentialSet();
            credentialSet.Load();
            var credentials = credentialSet
                .Where(c => c.Type == CredentialType.Generic)
                .Where(c => c.Target.StartsWith("vscode"))
                .ToList();
            foreach (var cred in credentials)
            {
                cred.Delete();
            }
        }
    }
}