#include "TFile.h"
#include "TBranch.h"
#include "TTree.h"
#include <dirent.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <sys/stat.h>

bool ends_with(std::string const &str, std::string const &suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return (str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0);
}

void preprocessing(const char *_input_path, const char *_output_path) {
    std::string input_path(_input_path);
    std::string output_path(_output_path);

    DIR *dir = opendir(input_path.c_str());
    struct dirent *entry = readdir(dir);
    while (entry != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            mkdir((output_path + "/" + entry->d_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            std::cout << entry->d_name << std::endl;
            std::string subpath = input_path + "/" + entry->d_name;
            std::cout << "\t" << subpath << std::endl;
            DIR *subdir = opendir(subpath.c_str());
            if (subdir == NULL) { std::cout << "\t" << "ERROR: " << errno << std::endl; }
            struct dirent *subentry = readdir(subdir);
            
            Int_t rfid = 0;
            while (subentry != NULL) {
               //std::cout << "\t\t" << subentry->d_name << " " << (int) subentry->d_type << std::endl;
                if (subentry->d_type != DT_DIR && ends_with(std::string(subentry->d_name), std::string(".root"))) {
                    std::string input_filepath = subpath + "/" + subentry->d_name;
                    std::cout << "\t\t" << input_filepath << std::endl;
                    TFile *f_raw = TFile::Open(input_filepath.c_str());
                    if (f_raw == NULL) {
                        rfid++;
                        subentry = readdir(subdir);
                        continue;
                    }
                    TTree *t_raw = (TTree *)f_raw->Get("ntINC2_0001100");
                    if (t_raw == NULL) {
                        delete f_raw;
                        rfid++;
                        subentry = readdir(subdir);
                        continue;
                    }
                    std::string output_filepath = output_path + "/" + entry->d_name + "/" + subentry->d_name;
                    TFile *f_new = TFile::Open(output_filepath.c_str(), "recreate");
                    if (f_new == NULL) {
                        delete f_raw;
                        rfid++;
                        subentry = readdir(subdir);
                        continue;
                    }
                    TTree *t_new = t_raw->CloneTree();
                    if (t_new == NULL) {
                        delete f_raw;
                        rfid++;
                        subentry = readdir(subdir);
                        continue;
                    }

                    TBranch *b_rfid = t_new->Branch("rfid", &rfid, "rfid/I");

                    Long64_t nentries = t_raw->GetEntries();

                    for (Long64_t i = 0; i < nentries; i++) {
                        t_raw->GetEntry(i);
                        b_rfid->Fill();
                    }
                    f_new->Write();
                    delete f_raw;
                    delete f_new;
                }
                rfid++;
                subentry = readdir(subdir);
            }
        }
        entry = readdir(dir);
    }
}
