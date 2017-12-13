/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid 

Source file: extras/Hadrons/VirtualMachine.hpp

Copyright (C) 2017

Author: Antonin Portelli <antonin.portelli@me.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
/*  END LEGAL */

#ifndef Hadrons_VirtualMachine_hpp_
#define Hadrons_VirtualMachine_hpp_

#include <Grid/Hadrons/Global.hpp>
#include <Grid/Hadrons/Graph.hpp>
#include <Grid/Hadrons/Environment.hpp>

BEGIN_HADRONS_NAMESPACE

#define DEFINE_VM_ALIAS \
inline VirtualMachine & vm(void) const\
{\
    return VirtualMachine::getInstance();\
}

/******************************************************************************
 *                   Virtual machine for module execution                     *
 ******************************************************************************/
// forward declaration of Module
class ModuleBase;

class VirtualMachine
{
    SINGLETON_DEFCTOR(VirtualMachine);
public:
    typedef SITE_SIZE_TYPE                       Size;
    typedef std::unique_ptr<ModuleBase>          ModPt;
    typedef  std::vector<std::set<unsigned int>> GarbageSchedule;
    struct MemoryPrint
    {
        Size         size;
        unsigned int module;
    };
    struct MemoryProfile
    {
        std::vector<std::map<unsigned int, Size>> module;
        std::vector<MemoryPrint>                  object;
    };
private:
    struct ModuleInfo
    {
        const std::type_info      *type{nullptr};
        std::string               name;
        ModPt                     data{nullptr};
        std::vector<unsigned int> input;
        size_t                    maxAllocated;
    };
public:
    // trajectory counter
    void                setTrajectory(const unsigned int traj);
    unsigned int        getTrajectory(void) const;
    // module management
    void                pushModule(ModPt &pt);
    template <typename M>
    void                createModule(const std::string name);
    template <typename M>
    void                createModule(const std::string name,
                                         const typename M::Par &par);
    void                createModule(const std::string name,
                                         const std::string type,
                                         XmlReader &reader);
    unsigned int        getNModule(void) const;
    ModuleBase *        getModule(const unsigned int address) const;
    ModuleBase *        getModule(const std::string name) const;
    template <typename M>
    M *                 getModule(const unsigned int address) const;
    template <typename M>
    M *                 getModule(const std::string name) const;
    unsigned int        getModuleAddress(const std::string name) const;
    std::string         getModuleName(const unsigned int address) const;
    std::string         getModuleType(const unsigned int address) const;
    std::string         getModuleType(const std::string name) const;
    std::string         getModuleNamespace(const unsigned int address) const;
    std::string         getModuleNamespace(const std::string name) const;
    bool                hasModule(const unsigned int address) const;
    bool                hasModule(const std::string name) const;
    // print VM content
    void                printContent(void) const;
    // module graph (could be a const reference if topoSort was const)
    Graph<unsigned int> getModuleGraph(void);
    // memory profile
    const MemoryProfile &getMemoryProfile(void);
    // garbage collector
    GarbageSchedule     makeGarbageSchedule(const std::vector<unsigned int> &p) const;
    // high-water memory function
    Size                memoryNeeded(const std::vector<unsigned int> &p,
                                     const GarbageSchedule &g);
    Size                memoryNeeded(const std::vector<unsigned int> &p);
    // general execution
    void                executeProgram(const std::vector<unsigned int> &p) const;
    void                executeProgram(const std::vector<std::string> &p) const;
private:
    // environment shortcut
    DEFINE_ENV_ALIAS;
    // module graph
    void makeModuleGraph(void);
    // memory profile
    void makeMemoryProfile(void);
    void resetProfile(void);
    void resizeProfile(void);
    void updateProfile(const unsigned int address);
    void cleanEnvironment(void);
    void memoryProfile(const std::string name);
    void memoryProfile(const unsigned int address);
private:
    // general
    unsigned int                        traj_;
    // module and related maps
    std::vector<ModuleInfo>             module_;
    std::map<std::string, unsigned int> moduleAddress_;
    std::string                         currentModule_{""};
    // module graph
    bool                                graphOutdated_{true};
    Graph<unsigned int>                 graph_;
    // memory profile
    bool                                memoryProfileOutdated_{true};
    MemoryProfile                       profile_;
};

/******************************************************************************
 *                   VirtualMachine template implementation                   *
 ******************************************************************************/
// module management ///////////////////////////////////////////////////////////
template <typename M>
void VirtualMachine::createModule(const std::string name)
{
    ModPt pt(new M(name));
    
    pushModule(pt);
}

template <typename M>
void VirtualMachine::createModule(const std::string name,
                               const typename M::Par &par)
{
    ModPt pt(new M(name));
    
    static_cast<M *>(pt.get())->setPar(par);
    pushModule(pt);
}

template <typename M>
M * VirtualMachine::getModule(const unsigned int address) const
{
    if (auto *pt = dynamic_cast<M *>(getModule(address)))
    {
        return pt;
    }
    else
    {
        HADRON_ERROR(Definition, "module '" + module_[address].name
                     + "' does not have type " + typeid(M).name()
                     + "(has type: " + getModuleType(address) + ")");
    }
}

template <typename M>
M * VirtualMachine::getModule(const std::string name) const
{
    return getModule<M>(getModuleAddress(name));
}

END_HADRONS_NAMESPACE

#endif // Hadrons_VirtualMachine_hpp_
