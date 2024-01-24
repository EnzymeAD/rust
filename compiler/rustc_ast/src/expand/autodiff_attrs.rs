use rustc_data_structures::stable_hasher::{HashStable, StableHasher};//, StableOrd};
use crate::HashStableContext;
use crate::expand::typetree::TypeTree;
use thin_vec::ThinVec;
//use rustc_expand::base::{Annotatable, ExtCtxt};
use std::str::FromStr;

use crate::NestedMetaItem;

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug)]
pub enum DiffMode {
    Inactive,
    Source,
    Forward,
    Reverse,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Eq, PartialEq, Encodable, Decodable, Debug)]
pub enum DiffActivity {
    None,
    Active,
    Const,
    Duplicated,
    DuplicatedNoNeed,
}
fn clause_diffactivity_discriminant(value: &DiffActivity) -> usize {
    match value {
        DiffActivity::None => 0,
        DiffActivity::Active => 1,
        DiffActivity::Const => 2,
        DiffActivity::Duplicated => 3,
        DiffActivity::DuplicatedNoNeed => 4,
    }
}
fn clause_diffmode_discriminant(value: &DiffMode) -> usize {
    match value {
        DiffMode::Inactive => 0,
        DiffMode::Source => 1,
        DiffMode::Forward => 2,
        DiffMode::Reverse => 3,
    }
}


impl<CTX: HashStableContext> HashStable<CTX> for DiffMode {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        clause_diffmode_discriminant(self).hash_stable(hcx, hasher);
    }
}

impl<CTX: HashStableContext> HashStable<CTX> for DiffActivity {
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        clause_diffactivity_discriminant(self).hash_stable(hcx, hasher);
    }
}

impl FromStr for DiffMode {
    type Err = ();

    fn from_str(s: &str) -> Result<DiffMode, ()> { match s {
            "Inactive" => Ok(DiffMode::Inactive),
            "Source" => Ok(DiffMode::Source),
            "Forward" => Ok(DiffMode::Forward),
            "Reverse" => Ok(DiffMode::Reverse),
            _ => Err(()),
        }
    }
}
impl FromStr for DiffActivity {
    type Err = ();

    fn from_str(s: &str) -> Result<DiffActivity, ()> { match s {
            "None" => Ok(DiffActivity::None),
            "Active" => Ok(DiffActivity::Active),
            "Const" => Ok(DiffActivity::Const),
            "Duplicated" => Ok(DiffActivity::Duplicated),
            "DuplicatedNoNeed" => Ok(DiffActivity::DuplicatedNoNeed),
            _ => Err(()),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AutoDiffAttrs {
    pub mode: DiffMode,
    pub ret_activity: DiffActivity,
    pub input_activity: Vec<DiffActivity>,
}

fn name(x: &NestedMetaItem) -> String {
    let segments = &x.meta_item().unwrap().path.segments;
    assert!(segments.len() == 1);
    segments[0].ident.name.to_string()
}

impl AutoDiffAttrs{
    pub fn has_ret_activity(&self) -> bool {
        match self.ret_activity {
            DiffActivity::None => false,
            _ => true,
        }
    }
    pub fn from_ast(meta_item: &ThinVec<NestedMetaItem>, has_ret: bool) -> Self {
        let mode = name(&meta_item[1]);
        let mode = DiffMode::from_str(&mode).unwrap();
        let activities: Vec<DiffActivity> = meta_item[2..].iter().map(|x| {
            let activity_str = name(&x);
            DiffActivity::from_str(&activity_str).unwrap()
        }).collect();

        // If a return type exist, we need to split the last activity,
        // otherwise we return None as placeholder.
        let (ret_activity, input_activity) = if has_ret {
            activities.split_last().unwrap()
        } else {
            (&DiffActivity::None, activities.as_slice())
        };

        AutoDiffAttrs {
            mode,
            ret_activity: *ret_activity,
            input_activity: input_activity.to_vec(),
        }
    }
}

//impl<CTX: HashStableContext> HashStable<CTX> for AutoDiffAttrs {
//    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
//        self.mode.hash_stable(hcx, hasher);
//        self.ret_activity.hash_stable(hcx, hasher);
//        self.input_activity.hash_stable(hcx, hasher);
//    }
//}

impl AutoDiffAttrs {
    pub fn inactive() -> Self {
        AutoDiffAttrs {
            mode: DiffMode::Inactive,
            ret_activity: DiffActivity::None,
            input_activity: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        match self.mode {
            DiffMode::Inactive => false,
            _ => {
                dbg!(&self);
                true
            },
        }
    }

    pub fn is_source(&self) -> bool {
        dbg!(&self);
        match self.mode {
            DiffMode::Source => true,
            _ => false,
        }
    }
    pub fn apply_autodiff(&self) -> bool {
        match self.mode {
            DiffMode::Inactive => false,
            DiffMode::Source => false,
            _ => {
                dbg!(&self);
                true
            },
        }
    }

    pub fn into_item(
        self,
        source: String,
        target: String,
        inputs: Vec<TypeTree>,
        output: TypeTree,
    ) -> AutoDiffItem {
        dbg!(&self);
        AutoDiffItem { source, target, inputs, output, attrs: self }
    }
}

#[derive(Clone, Eq, PartialEq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct AutoDiffItem {
    pub source: String,
    pub target: String,
    pub attrs: AutoDiffAttrs,
    pub inputs: Vec<TypeTree>,
    pub output: TypeTree,
}

//impl<CTX: HashStableContext> HashStable<CTX> for AutoDiffItem {
//    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
//        self.source.hash_stable(hcx, hasher);
//        self.target.hash_stable(hcx, hasher);
//        self.attrs.hash_stable(hcx, hasher);
//        for tt in &self.inputs {
//            tt.0.hash_stable(hcx, hasher);
//        }
//        //self.inputs.hash_stable(hcx, hasher);
//        self.output.0.hash_stable(hcx, hasher);
//    }
//}
