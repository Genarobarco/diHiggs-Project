#include "SampleAnalyzer/User/Analyzer/TemplateTaus.h"
using namespace MA5;
using namespace std;

// -----------------------------------------------------------------------------
// Initialize
// function called one time at the beginning of the analysis
// -----------------------------------------------------------------------------
bool TemplateTaus::Initialize(const MA5::Configuration& cfg, const std::map<std::string,std::string>& parameters)
{
  cout << "BEGIN Initialization" << endl;
  // initialize variables, histos
  cout << "END   Initialization" << endl;
  return true;
}

// -----------------------------------------------------------------------------
// Finalize
// function called one time at the end of the analysis
// -----------------------------------------------------------------------------
void TemplateTaus::Finalize(const SampleFormat& summary, const std::vector<SampleFormat>& files)
{
  cout << "BEGIN Finalization" << endl;
  // saving histos
  cout << "END   Finalization" << endl;
}

// -----------------------------------------------------------------------------
// Execute
// function called each time one event is read
// -----------------------------------------------------------------------------
bool TemplateTaus::Execute(SampleFormat& sample, const EventFormat& event)
{  
  MAfloat32 myWeight = 1.0;
  Manager()->InitializeForNewEvent(myWeight);
  
  std::vector<const RecJetFormat*> myjets;
  std::vector<const RecJetFormat*> mybjets;
  for(unsigned int ii=0; ii<event.rec()->jets().size(); ii++)
    {
      const RecJetFormat *CurrentJet = &(event.rec()->jets()[ii]);
      if(CurrentJet->pt()>20 and fabs(CurrentJet->eta())<2.5){
       if(CurrentJet->btag()){mybjets.push_back(CurrentJet);}
       else{myjets.push_back(CurrentJet);}
      }
    }
  SORTER->sort(myjets);
  SORTER->sort(mybjets);
  
  std::vector<const RecLeptonFormat*> myleptons;
  for(unsigned int ii=0; ii<event.rec()->electrons().size(); ii++)
   {
    const RecLeptonFormat *CurrentElec = &(event.rec()->electrons()[ii]);
    if(CurrentElec->pt()>10 and fabs(CurrentElec->eta()<2.5)){myleptons.push_back(CurrentElec);};
   }
  for(unsigned int ii=0; ii<event.rec()->muons().size(); ii++)
   {
    const RecLeptonFormat *CurrentMuon = &(event.rec()->muons()[ii]);
    if(CurrentMuon->pt()>10 and fabs(CurrentMuon->eta()<2.5)){myleptons.push_back(CurrentMuon);};
   }
   
   SORTER->sort(myleptons);
   
  std::vector<const RecTauFormat*> mytaus;
  for(unsigned int ii=0; ii<event.rec()->taus().size(); ii++)
    {
      const RecTauFormat *CurrentTau = &(event.rec()->taus()[ii]);
      if(CurrentTau->pt()>10 and fabs(CurrentTau->eta())<2.5){
      mytaus.push_back(CurrentTau);}
    }
    
    SORTER->sort(mytaus);
    
   std::vector<const RecTrackFormat*> mytracks;
  for(unsigned int ii=0; ii<event.rec()->tracks().size(); ii++)
    {
      const RecTrackFormat *CurrentTrack = &(event.rec()->tracks()[ii]);
      if(CurrentTrack->pt()>10 and fabs(CurrentTrack->eta())<2.5){
      mytracks.push_back(CurrentTrack);}
    }
    
    SORTER->sort(mytracks);
    
  unsigned int Nj = myjets.size();
  unsigned int Nb = mybjets.size();
  unsigned int Nl = myleptons.size();
  unsigned int Nta = mytaus.size();
  unsigned int Ntrk = mytracks.size();
  
   string folder = "path/";
   std::ofstream out0;
   out0.open(folder + "Nj.dat", std::ios::app);
   out0 << Nj << "\n";
   out0.close();   
   
   std::ofstream out1;
   out1.open(folder + "Nl.dat", std::ios::app);
   out1 << Nl << "\n";
   out1.close();
  
   std::ofstream out2;
   out2.open(folder + "Nta.dat", std::ios::app);
   out2 << Nta << "\n";
   out2.close();
   
   std::ofstream out3;
   out3.open(folder + "Ntrk.dat", std::ios::app);
   out3 << Ntrk << "\n";
   out3.close();
   
   std::ofstream out4;
   out4.open(folder + "Nb.dat", std::ios::app);
   out4 << Nb << "\n";
   out4.close();
  
  bool hadrcand = false;
  bool lepcand = false;
  bool semilepcand = false;
  bool bpair = false;
  
  unsigned int Nhadr = 0;
  unsigned int Nlep = 0;
  unsigned int Nslep = 0;
  
  if(Nb>=2){bpair=true;};
  if(Nta >= 2){hadrcand = true; Nhadr = Nhadr + 1;};
  if(Nta == 0 && Nl >=1 ){lepcand = true; Nlep = Nlep +1;};
  if(Nta == 1 && Nl >= 1 ){semilepcand = true; Nslep = Nslep +1;};
  
  double MET = event.rec()->MET().pt();
  
  if(hadrcand && bpair){
  
  double pTtau1 = mytaus[0]->momentum().Pt();
  double etatau1 = mytaus[0]->eta();
  double phitau1 = mytaus[0]->phi();
  double Mtau1 = mytaus[0]->momentum().M();
  
  double pTtau2 = mytaus[1]->momentum().Pt();
  double etatau2 = mytaus[1]->eta();
  double phitau2 = mytaus[1]->phi();
  double Mtau2 = mytaus[1]->momentum().M();
  
  double pTb1 = mybjets[0]->momentum().Pt();
  double etab1 = mybjets[0]->eta();
  double phib1 = mybjets[0]->phi();
  double Mb1 = mybjets[0]->momentum().M();
  
  double pTb2 = mybjets[1]->momentum().Pt();
  double etab2 = mybjets[1]->eta();
  double phib2 = mybjets[1]->phi();
  double Mb2 = mybjets[1]->momentum().M();

  double dRbb = mybjets[0]->dr(mybjets[1]);
  double pTbb = (mybjets[0]->momentum() + mybjets[1]->momentum()).Pt();
  double Mbb  = (mybjets[0]->momentum() + mybjets[1]->momentum()).M();
    
  double dRtata = mytaus[0]->dr(mytaus[1]);
  double pTtata = (mytaus[0]->momentum() + mytaus[1]->momentum()).Pt();
  double Mtata  = (mytaus[0]->momentum() + mytaus[1]->momentum()).M();
  
   string folder = "path/hadronic_ch/";
   std::ofstream out0;
   out0.open(folder + "MET.dat", std::ios::app);
   out0 << MET << "\n";
   out0.close();   
   
   std::ofstream out1;
   out1.open(folder + "pTtau1.dat", std::ios::app);
   out1 << pTtau1 << "\n";
   out1.close();
  
   std::ofstream out2;
   out2.open(folder + "etatau1.dat", std::ios::app);
   out2 << etatau1 << "\n";
   out2.close();
   
   std::ofstream out3;
   out3.open(folder + "phitau1.dat", std::ios::app);
   out3 << phitau1 << "\n";
   out3.close();
   
   std::ofstream out4;
   out4.open(folder + "Mtau1.dat", std::ios::app);
   out4 << Mtau1 << "\n";
   out4.close();
  
   std::ofstream out5;
   out5.open(folder + "pTtau2.dat", std::ios::app);
   out5 << pTtau2 << "\n";
   out5.close();
  
   std::ofstream out6;
   out6.open(folder + "etatau2.dat", std::ios::app);
   out6 << etatau2 << "\n";
   out6.close();
   
   std::ofstream out7;
   out7.open(folder + "phitau2.dat", std::ios::app);
   out7 << phitau2 << "\n";
   out7.close();
   
   std::ofstream out8;
   out8.open(folder + "Mtau2.dat", std::ios::app);
   out8 << Mtau2 << "\n";
   out8.close();
  
   std::ofstream out9;
   out9.open(folder + "dRtata.dat", std::ios::app);
   out9 << dRtata << "\n";
   out9.close();
   
   std::ofstream out10;
   out10.open(folder + "pTtata.dat", std::ios::app);
   out10 << pTtata << "\n";
   out10.close();
   
   std::ofstream out11;
   out11.open(folder + "Mtata.dat", std::ios::app);
   out11 << Mtata << "\n";
   out11.close();
   
    std::ofstream outb1;
   outb1.open(folder + "pTb1.dat", std::ios::app);
   outb1 << pTb1 << "\n";
   outb1.close();
  
   std::ofstream outb2;
   outb2.open(folder + "etab1.dat", std::ios::app);
   outb2 << etab1 << "\n";
   outb2.close();
   
   std::ofstream outb3;
   outb3.open(folder + "phib1.dat", std::ios::app);
   outb3 << phib1 << "\n";
   outb3.close();
   
   std::ofstream outb4;
   outb4.open(folder + "Mb1.dat", std::ios::app);
   outb4 << Mb1 << "\n";
   outb4.close();
  
   std::ofstream outb5;
   outb5.open(folder + "pTb2.dat", std::ios::app);
   outb5 << pTb2 << "\n";
   outb5.close();
  
   std::ofstream outb6;
   outb6.open(folder + "etab2.dat", std::ios::app);
   outb6 << etab2 << "\n";
   outb6.close();
   
   std::ofstream outb7;
   outb7.open(folder + "phib2.dat", std::ios::app);
   outb7 << phib2 << "\n";
   outb7.close();
   
   std::ofstream outb8;
   outb8.open(folder + "Mb2.dat", std::ios::app);
   outb8 << Mb2 << "\n";
   outb8.close();
  
   std::ofstream outb9;
   outb9.open(folder + "dRbb.dat", std::ios::app);
   outb9 << dRbb << "\n";
   outb9.close();
   
   std::ofstream outb10;
   outb10.open(folder + "pTbb.dat", std::ios::app);
   outb10 << pTbb << "\n";
   outb10.close();
   
   std::ofstream outb11;
   outb11.open(folder + "Mbb.dat", std::ios::app);
   outb11 << Mbb << "\n";
   outb11.close();
    };
  if(semilepcand){
  
  double pTtau1 = mytaus[0]->momentum().Pt();
  double etatau1 = mytaus[0]->eta();
  double phitau1 = mytaus[0]->phi();
  double Mtau1 = mytaus[0]->momentum().M();
  
  double pTlep1 = myleptons[1]->momentum().Pt();
  double etalep1 = myleptons[1]->eta();
  double philep1 = myleptons[1]->phi();
  double Mlep1 = myleptons[1]->momentum().M();
  
  double dRtalep = mytaus[0]->dr(myleptons[0]);
  double pTtalep = (mytaus[0]->momentum() + myleptons[0]->momentum()).Pt();
  double Mtalep  = (mytaus[0]->momentum() + myleptons[0]->momentum()).M();
  
  double pTb1 = mybjets[0]->momentum().Pt();
  double etab1 = mybjets[0]->eta();
  double phib1 = mybjets[0]->phi();
  double Mb1 = mybjets[0]->momentum().M();
  
  double pTb2 = mybjets[1]->momentum().Pt();
  double etab2 = mybjets[1]->eta();
  double phib2 = mybjets[1]->phi();
  double Mb2 = mybjets[1]->momentum().M();

  double dRbb = mybjets[0]->dr(mybjets[1]);
  double pTbb = (mybjets[0]->momentum() + mybjets[1]->momentum()).Pt();
  double Mbb  = (mybjets[0]->momentum() + mybjets[1]->momentum()).M();

  
   string folder = "path/semileptonic_ch/";
   std::ofstream out0;
   out0.open(folder + "MET.dat", std::ios::app);
   out0 << MET << "\n";
   out0.close();   
   
   std::ofstream out1;
   out1.open(folder + "pTtau1.dat", std::ios::app);
   out1 << pTtau1 << "\n";
   out1.close();
  
   std::ofstream out2;
   out2.open(folder + "etatau1.dat", std::ios::app);
   out2 << etatau1 << "\n";
   out2.close();
   
   std::ofstream out3;
   out3.open(folder + "phitau1.dat", std::ios::app);
   out3 << phitau1 << "\n";
   out3.close();
   
   std::ofstream out4;
   out4.open(folder + "Mtau1.dat", std::ios::app);
   out4 << Mtau1 << "\n";
   out4.close();
  
   std::ofstream out5;
   out5.open(folder + "pTlep1.dat", std::ios::app);
   out5 << pTlep1 << "\n";
   out5.close();
  
   std::ofstream out6;
   out6.open(folder + "etalep1.dat", std::ios::app);
   out6 << etalep1 << "\n";
   out6.close();
   
   std::ofstream out7;
   out7.open(folder + "philep1.dat", std::ios::app);
   out7 << philep1 << "\n";
   out7.close();
   
   std::ofstream out8;
   out8.open(folder + "Mlep1.dat", std::ios::app);
   out8 << Mlep1 << "\n";
   out8.close();
  
   std::ofstream out9;
   out9.open(folder + "dRtalep.dat", std::ios::app);
   out9 << dRtalep << "\n";
   out9.close();
   
   std::ofstream out10;
   out10.open(folder + "pTtalep.dat", std::ios::app);
   out10 << pTtalep << "\n";
   out10.close();
   
   std::ofstream out11;
   out11.open(folder + "Mtalep.dat", std::ios::app);
   out11 << Mtalep << "\n";
   out11.close();
   
    std::ofstream outb1;
   outb1.open(folder + "pTb1.dat", std::ios::app);
   outb1 << pTb1 << "\n";
   outb1.close();
  
   std::ofstream outb2;
   outb2.open(folder + "etab1.dat", std::ios::app);
   outb2 << etab1 << "\n";
   outb2.close();
   
   std::ofstream outb3;
   outb3.open(folder + "phib1.dat", std::ios::app);
   outb3 << phib1 << "\n";
   outb3.close();
   
   std::ofstream outb4;
   outb4.open(folder + "Mb1.dat", std::ios::app);
   outb4 << Mb1 << "\n";
   outb4.close();
  
   std::ofstream outb5;
   outb5.open(folder + "pTb2.dat", std::ios::app);
   outb5 << pTb2 << "\n";
   outb5.close();
  
   std::ofstream outb6;
   outb6.open(folder + "etab2.dat", std::ios::app);
   outb6 << etab2 << "\n";
   outb6.close();
   
   std::ofstream outb7;
   outb7.open(folder + "phib2.dat", std::ios::app);
   outb7 << phib2 << "\n";
   outb7.close();
   
   std::ofstream outb8;
   outb8.open(folder + "Mb2.dat", std::ios::app);
   outb8 << Mb2 << "\n";
   outb8.close();
  
   std::ofstream outb9;
   outb9.open(folder + "dRbb.dat", std::ios::app);
   outb9 << dRbb << "\n";
   outb9.close();
   
   std::ofstream outb10;
   outb10.open(folder + "pTbb.dat", std::ios::app);
   outb10 << pTbb << "\n";
   outb10.close();
   
   std::ofstream outb11;
   outb11.open(folder + "Mbb.dat", std::ios::app);
   outb11 << Mbb << "\n";
   outb11.close();
  };
  if(lepcand){
  
  double pTlep1 = myleptons[1]->momentum().Pt();
  double etalep1 = myleptons[1]->eta();
  double philep1 = myleptons[1]->phi();
  double Mlep1 = myleptons[1]->momentum().M();
  
  double pTb1 = mybjets[0]->momentum().Pt();
  double etab1 = mybjets[0]->eta();
  double phib1 = mybjets[0]->phi();
  double Mb1 = mybjets[0]->momentum().M();
  
  double pTb2 = mybjets[1]->momentum().Pt();
  double etab2 = mybjets[1]->eta();
  double phib2 = mybjets[1]->phi();
  double Mb2 = mybjets[1]->momentum().M();

  double dRbb = mybjets[0]->dr(mybjets[1]);
  double pTbb = (mybjets[0]->momentum() + mybjets[1]->momentum()).Pt();
  double Mbb  = (mybjets[0]->momentum() + mybjets[1]->momentum()).M();
  
   string folder = "path/leptonic_ch/";
   std::ofstream out0;
   out0.open(folder + "pTlep1.dat", std::ios::app);
   out0 << pTlep1 << "\n";
   out0.close();
  
   std::ofstream out1;
   out1.open(folder + "etalep1.dat", std::ios::app);
   out1 << etalep1 << "\n";
   out1.close();
   
   std::ofstream out2;
   out2.open(folder + "philep1.dat", std::ios::app);
   out2 << philep1 << "\n";
   out2.close();
   
   std::ofstream out3;
   out3.open(folder + "Mlep1.dat", std::ios::app);
   out3 << Mlep1 << "\n";
   out3.close();
   
    std::ofstream outb1;
   outb1.open(folder + "pTb1.dat", std::ios::app);
   outb1 << pTb1 << "\n";
   outb1.close();
  
   std::ofstream outb2;
   outb2.open(folder + "etab1.dat", std::ios::app);
   outb2 << etab1 << "\n";
   outb2.close();
   
   std::ofstream outb3;
   outb3.open(folder + "phib1.dat", std::ios::app);
   outb3 << phib1 << "\n";
   outb3.close();
   
   std::ofstream outb4;
   outb4.open(folder + "Mb1.dat", std::ios::app);
   outb4 << Mb1 << "\n";
   outb4.close();
  
   std::ofstream outb5;
   outb5.open(folder + "pTb2.dat", std::ios::app);
   outb5 << pTb2 << "\n";
   outb5.close();
  
   std::ofstream outb6;
   outb6.open(folder + "etab2.dat", std::ios::app);
   outb6 << etab2 << "\n";
   outb6.close();
   
   std::ofstream outb7;
   outb7.open(folder + "phib2.dat", std::ios::app);
   outb7 << phib2 << "\n";
   outb7.close();
   
   std::ofstream outb8;
   outb8.open(folder + "Mb2.dat", std::ios::app);
   outb8 << Mb2 << "\n";
   outb8.close();
  
   std::ofstream outb9;
   outb9.open(folder + "dRbb.dat", std::ios::app);
   outb9 << dRbb << "\n";
   outb9.close();
   
   std::ofstream outb10;
   outb10.open(folder + "pTbb.dat", std::ios::app);
   outb10 << pTbb << "\n";
   outb10.close();
   
   std::ofstream outb11;
   outb11.open(folder + "Mbb.dat", std::ios::app);
   outb11 << Mbb << "\n";
   outb11.close();
  };
  
 return true;
}

